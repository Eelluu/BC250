#!/usr/bin/env bash
# rocm-bc250-setup.sh - ROCm/HIP en el AMD BC-250 (gfx1013) sobre Alpine, via contenedor glibc.
#
# Honestidad primero: esto NO es un instalador magico. La receta completa (y verificada)
# es de https://github.com/akandr/bc250-rocm sobre Fedora 44 + ROCm 7.1.1, y tiene 8 pasos:
# el lado kernel (amdkfd parcheado + unlock de 40 CU), firmware SDMA, politica de reloj,
# rocBLAS nativo gfx1013 (~50 min), fix de comgr, overrides de librerias, 13 parches a
# llama.cpp y verificacion. Este script hace tres cosas:
#
#   1) DIAGNOSTICA el host Alpine (paso imprescindible: si el kernel no esta listo, no sigas)
#   2) MONTA el contenedor Fedora con /dev/kfd + /dev/dri (la via sensata en musl)
#   3) LANZA los pasos automatizables de la receta, uno a uno, sobre tu llamada
#
# Uso:
#   ./rocm-bc250-setup.sh check       # solo diagnostica el host. No toca nada.
#   ./rocm-bc250-setup.sh up          # instala podman y crea el contenedor
#   ./rocm-bc250-setup.sh recreate    # re-crea el contenedor (p.ej. para montar los modelos)
#   ./rocm-bc250-setup.sh deps        # paquetes de build dentro del contenedor
#   ./rocm-bc250-setup.sh rocblas     # rocBLAS nativo para gfx1013  (~50 min)
#   ./rocm-bc250-setup.sh comgr       # parchea libamd_comgr (VGPRs de gfx10)
#   ./rocm-bc250-setup.sh overrides   # pone rocBLAS+comgr por delante del sistema
#   ./rocm-bc250-setup.sh llamacpp    # clona llama.cpp (commit 7ba604f) y aplica los 13 parches
#   ./rocm-bc250-setup.sh gate        # test-backend-ops sobre ROCm0 (valida la serie de parches)
#   ./rocm-bc250-setup.sh verify      # rocminfo + comprobaciones de la receta
#   ./rocm-bc250-setup.sh shell       # shell dentro del contenedor
#   ./rocm-bc250-setup.sh bench       # lanza el comparador ROCm vs Vulkan (script aparte)
#   ./rocm-bc250-setup.sh down        # borra el contenedor (deja el trabajo en /opt/rocm-bc250)
#
# Requisitos del host: kernel con los parches de amdkfd (tu arieltune ya trae 21/24/25/26),
# 40 CU enrutados, y ~25 GB libres. El script te lo dice en `check`.
set -euo pipefail

IMAGE="${BC250_ROCM_IMAGE:-docker.io/library/fedora:44}"
NAME="${BC250_ROCM_NAME:-bc250-rocm}"
HOST_WORK="${BC250_ROCM_WORK:-/opt/rocm-bc250}"      # persistente en el host
CTR_WORK=/root/bc250                                  # dentro del contenedor
REPO_URL="https://github.com/akandr/bc250-rocm.git"
ROCM_VER="${BC250_ROCM_VER:-rocm-7.1.1}"
# Directorio de modelos del host que se monta DENTRO del contenedor (misma ruta),
# para que llama-bench de ROCm pueda leerlos. Sobrescribe con BC250_MODELS_DIR.
MODELS_DIR="${BC250_MODELS_DIR:-/opt/llama-models}"
# Base de llama.cpp que midio akandr: los 13 parches son para este commit exacto.
# Sobre master el 0001 ya esta mergeado y el resto no aplica.
LLAMACPP_COMMIT="${BC250_LLAMACPP_COMMIT:-7ba604f1cb61cd14898138e9abc0b4ff2601f180}"

c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_off=$'\033[0m'
ok()   { echo "  ${c_ok}OK${c_off}    $*"; }
warn() { echo "  ${c_warn}WARN${c_off}  $*"; }
bad()  { echo "  ${c_err}FAIL${c_off}  $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
need_root() { [ "$(id -u)" = 0 ] || die "requiere root"; }

# ---------------------------------------------------------------- check

param() { cat "/sys/module/amdgpu/parameters/$1" 2>/dev/null || echo ""; }

check_host() {
	local fails=0
	echo "== Lado kernel (lo que la receta exige) =="

	if [ -e /dev/kfd ]; then ok "/dev/kfd presente"; else bad "/dev/kfd NO existe: falta KFD (amdkfd). Sin esto no hay HIP."; fails=$((fails + 1)); fi
	if [ -e /dev/dri/renderD128 ]; then ok "/dev/dri/renderD128 presente"; else bad "no hay nodo render de DRM"; fails=$((fails + 1)); fi

	local fp fbr sp sfw ssd
	fp="$(param bc250_flush_pasid_kiq)"
	fbr="$(param bc250_flush_by_runlist)"
	sp="$(param sched_policy)"
	sfw="$(param bc250_sdma_fw)"
	ssd="$(param bc250_skip_sdma0)"

	# Parche 25 de arieltune: booleano (0/1) y solo en el lado *unmap*.
	# bc250-rocm lo extendio al lado *map* (su valor 3 = 1|2); arieltune no lo trae.
	if [ -n "$fbr" ]; then
		if [ "$fbr" != "0" ]; then ok "bc250_flush_by_runlist=$fbr (armado)"
		else bad "bc250_flush_by_runlist=0: ponlo a 1 en el cmdline."; fails=$((fails + 1)); fi
		warn "nota: el parche 25 de arieltune es un booleano y solo hace el flush en *unmap*; bc250-rocm necesita ademas el de *map*. Fallos con churn de memoria = esto."
	else
		bad "no existe bc250_flush_by_runlist: falta el parche 25 de arieltune en tu kernel."; fails=$((fails + 1))
	fi

	# arieltune (parche 26) implementa el override; Alpine trae el blob .zst, que vale igual.
	if [ -n "$sfw" ]; then
		if [ "$sfw" = "navi12" ]; then ok "bc250_sdma_fw=navi12 (microcodigo SDMA de navi12)"
		else warn "bc250_sdma_fw=$sfw (la receta usa navi12)"; fi
	else
		warn "bc250_sdma_fw vacio: para ROCm arranca con amdgpu.bc250_sdma_fw=navi12"
	fi
	if ls /lib/firmware/amdgpu/navi12_sdma* >/dev/null 2>&1; then
		ok "blobs SDMA navi12 presentes: $(ls /lib/firmware/amdgpu/ | grep -i '^navi12_sdma' | tr '\n' ' ')"
	else
		bad "faltan navi12_sdma*.bin[.zst] en /lib/firmware/amdgpu/ -> apk add linux-firmware-amdgpu"; fails=$((fails + 1))
	fi

	# El parche 19 (bc250_skip_sdma0) esta RETIRADO: lo sustituyen 26+27.
	if [ -n "$ssd" ]; then
		ok "bc250_skip_sdma0=$ssd presente (parche 19, retirado: solo si reviertes el firmware de navi12)"
	else
		ok "(info) sin bc250_skip_sdma0: correcto, el parche 19 esta retirado y lo sustituyen 26+27"
	fi
	# Parche 27: escribe TRAP_ENABLE de SDMA en gfx_resume.
	if [ -n "$(param bc250_early_sdma_trap)" ]; then
		if [ "$(param bc250_early_sdma_trap)" = "1" ]; then ok "bc250_early_sdma_trap=1"
		else warn "bc250_early_sdma_trap=$(param bc250_early_sdma_trap) (la receta usa 1)"; fi
	else
		warn "sin bc250_early_sdma_trap: los fences de SDMA0 caen al temporizador de 500 ms en cada arranque"
	fi

	# Este SI es de los parches de bc250-rocm; que no exista aqui no es un fallo.
	if [ -n "$fp" ]; then
		if [ "$fp" = "0" ]; then ok "bc250_flush_pasid_kiq=0 (flush corregido)"
		else warn "bc250_flush_pasid_kiq=$fp (bc250-rocm usa 0)"; fi
	else
		ok "(info) tu kernel no expone bc250_flush_pasid_kiq: es de los parches de bc250-rocm, no de arieltune"
	fi

	if [ "$sp" = "2" ]; then bad "amdgpu.sched_policy=2: con el flush corregido esto wedgea bajo compute sostenido. Quitalo del cmdline."; fails=$((fails + 1))
	else ok "sched_policy no es 2 (${sp:-sin definir})"; fi

	local cc; cc="$(param bc250_cc_write_mode)"
	if [ "$cc" = "3" ]; then ok "bc250_cc_write_mode=3 (liberacion de 40 CU armada)"
	elif [ -z "$cc" ]; then warn "sin bc250_cc_write_mode: estaras a 24 CU (funciona, pero rindes menos)"
	else warn "bc250_cc_write_mode=$cc"; fi

	if have arieltune; then
		local m; m="$(arieltune apu cu status 2>/dev/null | grep -i 'masks:' | head -1 | sed 's/^ *//')"
		if [ -n "$m" ]; then ok "ruta de CU: $m"; fi
	fi

	echo
	echo "  Cmdline que espera la receta (solo lo que falte de arriba):"
	echo "    amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_by_runlist=1 \\"
	echo "    amdgpu.bc250_sdma_fw=navi12 amdgpu.bc250_early_sdma_trap=1"
	echo "    (y NO poner amdgpu.sched_policy=2; bc250_skip_sdma0 es del parche 19, retirado)"

	echo
	echo "== Recursos del host =="
	local libre; libre="$(df -Pk / | awk 'NR==2{printf "%.1f", $4/1048576}')"
	if awk "BEGIN{exit !($libre >= 25)}"; then ok "${libre} GB libres en / (recomendado >=25: rootfs Fedora + build de rocBLAS)"
	else warn "solo ${libre} GB libres en /; el contenedor + rocBLAS pueden necesitar ~25 GB"; fi
	if have podman; then ok "podman instalado"; else warn "podman no instalado (lo hace 'up')"; fi

	echo
	if [ "$fails" -eq 0 ]; then
		echo "${c_ok}Host listo${c_off} para el lado usuario. Sigue con: $0 up"
	else
		echo "${c_err}$fails comprobacion(es) del kernel fallan${c_off}: el lado usuario no arregla esto."
		echo "Referencia: https://github.com/akandr/bc250-rocm (pasos 1-3). Tu arieltune ya trae los parches 21/24/25/26; revisa el cmdline (sched_policy, bc250_flush_by_runlist=3)."
		exit 1
	fi
}

# ---------------------------------------------------------------- contenedor

ensure_podman() {
	if have podman; then return 0; fi
	echo "Instalando podman..."
	apk add --no-cache podman crun fuse-overlayfs
	rc-update add cgroups default 2>/dev/null || true
	rc-service cgroups start 2>/dev/null || true
	rc-service podman start 2>/dev/null || true
}

ctr_exists() { podman container exists "$NAME" 2>/dev/null; }

up() {
	need_root
	ensure_podman
	mkdir -p "$HOST_WORK"
	if ctr_exists; then echo "El contenedor '$NAME' ya existe."; return 0; fi
	echo "Creando contenedor '$NAME' desde $IMAGE ..."
	podman pull "$IMAGE"
	# --device para los nodos de computo; seccomp/label relajados porque ROCm hace
	# operaciones que la politica por defecto bloquea (ptrace, ioctls de KFD).
	# --network=host: dnf y git necesitan red y evita el setup de netavark en Alpine.
	# El directorio de modelos se monta en la MISMA ruta para que llama-bench los vea.
	local vol=(-v "$HOST_WORK:$CTR_WORK")
	if [ -d "$MODELS_DIR" ]; then
		vol+=(-v "$MODELS_DIR:$MODELS_DIR:ro")
	else
		warn "$MODELS_DIR no existe en el host: el contenedor no vera los modelos."
		warn "Si los tienes en otro sitio: export BC250_MODELS_DIR=/ruta y re-crea el contenedor."
	fi
	podman run -d --name "$NAME" \
		--network=host --ipc=host \
		--device /dev/kfd --device /dev/dri \
		--security-opt seccomp=unconfined --security-opt label=disable \
		"${vol[@]}" \
		"$IMAGE" sleep infinity
	echo "Listo. Comprueba con: $0 shell   (y dentro: ls /dev/kfd /dev/dri $MODELS_DIR)"
}

# Re-crea el contenedor para anadirle montajes (p.ej. el de modelos) sin perder
# el trabajo largo: llama.cpp vive en $HOST_WORK (bind mount) y rocBLAS se mueve
# ahi antes de borrar. Los paquetes ROCm si se pierden -> hay que re-hacer 'deps'.
recreate() {
	need_root
	require_ctr
	echo "Re-creando '$NAME' con los montajes actuales."
	if exe bash -lc "[ -d $CTR_WORK/rb711 ]"; then
		echo "  rocBLAS ya esta en $HOST_WORK/rb711 (persistente)"
	elif exe bash -lc '[ -d /root/rb711 ] && [ ! -L /root/rb711 ]'; then
		echo "  moviendo rocBLAS a $CTR_WORK/rb711 (persistente) para no recompilarlo..."
		exe bash -lc "mv /root/rb711 $CTR_WORK/rb711"
		# Abortar ANTES de borrar nada si no hemos podido preservarlo: perderlo
		# son ~50 minutos de build.
		exe bash -lc "[ -f $CTR_WORK/rb711/install/lib64/librocblas.so.5 ]" \
			|| die "no he podido preservar rocBLAS ($HOST_WORK/rb711/install/lib64/librocblas.so.5). NO borro el contenedor."
	elif exe bash -lc '[ -L /root/rb711 ]'; then
		echo "  rocBLAS ya estaba enlazado a $CTR_WORK/rb711"
	else
		warn "no encuentro rocBLAS: tras recrear habra que re-hacer '$0 rocblas' (~50 min)"
	fi
	podman rm -f "$NAME"
	up
	# El symlink /root/rb711 vive en el rootfs del contenedor, asi que hay que
	# rehacerlo en el nuevo (el dato esta a salvo en el bind mount).
	if exe bash -lc "[ -d $CTR_WORK/rb711 ]"; then
		exe bash -lc "ln -sfn $CTR_WORK/rb711 /root/rb711"
		echo "  symlink /root/rb711 -> $CTR_WORK/rb711 recreado"
	fi
	echo "Los paquetes ROCm se fueron con el contenedor: reinstalando (deps)..."
	deps
	comgr
	overrides
	echo "Listo. llama.cpp NO se recompila (vive en $CTR_WORK/llama.cpp)."
}

exe() { podman exec "$NAME" "$@"; }
require_ctr() { if ! ctr_exists; then die "no hay contenedor: ejecuta primero '$0 up'"; fi; }
require_deps() {
	require_ctr
	if ! exe bash -lc '[ -d /root/bc250/bc250-rocm ] && command -v git >/dev/null && command -v python3 >/dev/null && command -v patch >/dev/null'; then
		die "faltan dependencias en el contenedor (git/python3/patch o el repo bc250-rocm): ejecuta primero '$0 deps'"
	fi
}
shell() { need_root; ctr_exists || die "no hay contenedor; ejecuta '$0 up'"; podman exec -it "$NAME" bash; }
down() { need_root; if ctr_exists; then podman rm -f "$NAME"; fi; echo "Contenedor borrado. El trabajo sigue en $HOST_WORK"; }

clone_repo() {
	exe bash -lc '
		set -e
		mkdir -p /root/bc250 && cd /root/bc250
		[ -d bc250-rocm ] || git clone --depth 1 '"$REPO_URL"' bc250-rocm
		cd bc250-rocm && echo "bc250-rocm en $(git rev-parse --short HEAD)"'
}

deps() {
	require_ctr
	echo "Instalando ROCm + dependencias de build (paso 4 de la receta)..."
	# PRIMERO los paquetes: la imagen fedora:44 no trae ni git ni python3, y
	# clone_repo() los necesita.
	# hipblas-devel NO esta en la lista del README de bc250-rocm, pero el
	# CMakeLists de ggml-hip hace find_package(hipblas REQUIRED): sin el, el
	# configure de llama.cpp muere. (rccl solo se pide con GGML_HIP_RCCL.)
	exe bash -lc 'set -e; dnf -y install \
		rocm-hip-devel hipblas-devel rocblas-devel rocm-cmake rocminfo roctracer-devel msgpack-devel \
		ninja-build ccache python3-pyyaml python3-msgpack python3-joblib git patch cmake clang gcc-c++ make which'
	echo "Clonando la receta bc250-rocm..."
	clone_repo
	echo "Nota: si Fedora trae ROCm 7.2.x en vez de 7.1.1, la receta de rocBLAS/comgr esta escrita para 7.1.1."
}

rocblas() {
	require_deps
	local rb=/root/rb711
	if exe bash -lc "[ -d $rb/install/lib64/rocblas/library ] && ls $rb/install/lib64/rocblas/library/ | grep -q gfx1013"; then
		echo "rocBLAS ya construido para gfx1013. Code objects:"
		exe bash -lc "ls $rb/install/lib64/rocblas/library/ | grep -c gfx1013"
		return 0
	fi
	echo "rocBLAS nativo para gfx1013. Son ~50 minutos en esta placa."
	exe bash -lc "set -e
		mkdir -p $rb && cd $rb
		[ -d rocm-libraries ] || git clone --depth 1 --filter=blob:none --sparse -b $ROCM_VER https://github.com/ROCm/rocm-libraries.git
		git -C rocm-libraries sparse-checkout set projects/rocblas shared/tensile
		python3 /root/bc250/bc250-rocm/scripts/apply_gfx1013_rocblas711.py rocm-libraries
		/root/bc250/bc250-rocm/scripts/build_rocblas711_gfx1013.sh $rb
		echo
		echo 'code objects gfx1013 en la libreria:'
		ls $rb/install/lib64/rocblas/library/ | grep -c gfx1013"
}

comgr() {
	require_deps
	exe bash -lc "set -e
		python3 /root/bc250/bc250-rocm/scripts/fix_comgr_gfx10_vgprs.py \
			/usr/lib64/libamd_comgr.so.3.0 /root/libamd_comgr.so.3"
}

overrides() {
	require_deps
	exe bash -lc "set -e
		/root/bc250/bc250-rocm/scripts/install_rocm711_overrides.sh /root/rb711/install /root/libamd_comgr.so.3
		ldconfig
		ldconfig -p | grep -E 'librocblas.so.5 |libamd_comgr.so.3 ' || true"
}

llamacpp() {
	require_deps
	exe bash -lc "set -e
		cd /root/bc250
		if [ -x llama.cpp/build-hip/bin/llama-bench ]; then echo 'ya construido'; exit 0; fi
		rm -rf llama.cpp
		mkdir llama.cpp && cd llama.cpp
		git init -q
		git remote add origin https://github.com/ggml-org/llama.cpp
		git fetch -q --depth 1 origin $LLAMACPP_COMMIT
		git checkout -q -f FETCH_HEAD
		for p in /root/bc250/bc250-rocm/patches/llamacpp/*.patch; do
			echo \"aplicando \$(basename \$p)\"
			# El 0008 lleva una linea de contexto (getenv BC250_PATH) que era
			# instrumentacion local del autor y no esta en la base: git apply
			# exige el contexto entero, asi que se cae a GNU patch con fuzz.
			# Con fuzz conviene validar despues: '$0 gate'.
			git apply --ignore-whitespace \"\$p\" 2>/dev/null || patch -p1 -F3 --no-backup-if-mismatch -i \"\$p\" || { echo 'FALLO al aplicar' \"\$p\"; exit 1; }
		done
		GOMP=\$(ls -d /usr/lib/gcc/x86_64-redhat-linux/*/libgomp.so 2>/dev/null | tail -1)
		cmake -S . -B build-hip -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1013 \
			-DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
			-DOpenMP_gomp_LIBRARY=\$GOMP -DOpenMP_pthread_LIBRARY=/usr/lib64/libpthread.a
		# El aviso -Wpass-failed del kernel packed-fp16 de RDNA1 se repite por cada
		# arquitectura; se filtra para que la build se lea. Los errores si pasan.
		set +e
		cmake --build build-hip -j6 --target llama-bench llama-cli llama-perplexity test-backend-ops 2>&1 \\
			| tee /root/bc250/build.log \\
			| grep -E --line-buffered '%]|error|Error|FAILED|undefined reference'
		rc=\${PIPESTATUS[0]}
		set -e
		[ \$rc -eq 0 ] || { echo 'la build fallo; log completo: /root/bc250/build.log'; exit 1; }"
}

gate() {
	require_deps
	if ! exe bash -lc '[ -x /root/bc250/llama.cpp/build-hip/bin/test-backend-ops ]'; then
		die "test-backend-ops no construido: ejecuta primero '$0 llamacpp'"
	fi
	echo "Gate de correccion: test-backend-ops sobre ROCm0."
	echo "El 0008 se aplica con fuzz, asi que esto es lo que lo valida (la receta espera todo OK)."
	exe bash -lc 'cd /root/bc250/llama.cpp && ./build-hip/bin/test-backend-ops -b ROCm0 2>&1 | tail -20'
}

verify() {
	require_ctr
	if ! exe bash -lc 'command -v rocminfo >/dev/null 2>&1'; then
		echo "El contenedor todavia no tiene ROCm instalado."
		echo "Orden correcto:"
		echo "  $0 up  ->  $0 deps  ->  $0 rocblas  ->  $0 comgr  ->  $0 overrides  ->  $0 llamacpp  ->  $0 verify"
		return 0
	fi
	echo "== rocminfo =="
	exe bash -lc 'rocminfo 2>/dev/null | grep -iE "Name:|gfx1013|Marketing" | head -20 || echo "rocminfo no disponible"'
	echo
	echo "== code objects gfx1013 en rocBLAS =="
	exe bash -lc 'ls /root/rb711/install/lib64/rocblas/library/ 2>/dev/null | grep -c gfx1013 || echo 0'
	echo
	echo "== llama.cpp ROCm =="
	exe bash -lc 'ls -la /root/bc250/llama.cpp/build-hip/bin/llama-bench 2>/dev/null || echo "no construido (ejecuta: $0 llamacpp)"'
	echo
	echo "Si rocminfo muestra gfx1013 y el contador de rocBLAS es > 0, el stack esta en pie."
	echo "Recuerda: el reset de GPU NO funciona en este chip; un fallo de pagina sostenido deja la GPU inservible hasta reiniciar."
}

bench() {
	local here; here="$(cd "$(dirname "$0")/.." && pwd)"
	[ -x "$here/benchmarks/bench-rocm-vs-vulkan.sh" ] || die "no encuentro benchmarks/bench-rocm-vs-vulkan.sh"
	"$here/benchmarks/bench-rocm-vs-vulkan.sh" --rocm "podman exec $NAME /root/bc250/llama.cpp/build-hip/bin/llama-bench" "$@"
}

case "${1:-}" in
	check)     check_host ;;
	up)        up ;;
	recreate)  recreate ;;
	deps)      deps ;;
	rocblas)   rocblas ;;
	comgr)     comgr ;;
	overrides) overrides ;;
	llamacpp)  llamacpp ;;
	gate)      gate ;;
	verify)    verify ;;
	shell)     shell ;;
	bench)     shift; bench "$@" ;;
	down)      down ;;
	*)         sed -n '2,30p' "$0"; exit 1 ;;
esac
