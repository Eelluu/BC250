#!/usr/bin/env python3
"""web_search.py - one-shot web search for the Discord bot's chat.

The chat prompt used to tell the model to run `ddgr`, which scrapes DuckDuckGo
and gets throttled from a residential IP - the same reason music-discover managed
to fail silently for two months. This is what the prompt should point at instead:
it goes through the local SearXNG and falls back to ddgr, reusing the research
agent's own _web_search so both paths behave identically.

Usage: python3 /opt/netscan/web_search.py "query" [max_results]
"""
import sys

sys.path.insert(0, "/opt/netscan")

from research_agent import _web_search  # noqa: E402


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python3 /opt/netscan/web_search.py "query" [max_results]')
        return 2
    query = sys.argv[1].strip()
    limit = 5
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        limit = max(1, min(10, int(sys.argv[2])))
    print(_web_search(query, max_results=limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
