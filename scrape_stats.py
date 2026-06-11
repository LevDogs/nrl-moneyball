"""
Scrape NRL team stats from nrl.com for all 17 teams.
Extracts q-data from server-rendered HTML pages.
Outputs a Python dict ready to paste into app.py.
"""
import json
import re
import subprocess
import sys
import time

STATS = {
    76: "Points",
    38: "Tries",
    30: "Linebreaks",
    29: "TackleBreaks",
    1000112: "PCM",
    35: "TryAssists",
    28: "Offloads",
    1000037: "RunMetres",
    1000038: "AllRuns",
    3: "Tackles",
    4: "MissedTackles",
    1000003: "IneffTackles",
    37: "Errors",
    32: "KickMetres",
}

BASE_URL = "https://www.nrl.com/stats/teams/"


def fetch_page(stat_id):
    url = f"{BASE_URL}?competition=111&season=2026&stat={stat_id}"
    try:
        result = subprocess.run(
            ["curl", "-s", "-L",
             "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
             "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
             "-H", "Accept-Language: en-US,en;q=0.9",
             url],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except Exception as e:
        print(f"  Error fetching stat {stat_id}: {e}", file=sys.stderr)
        return None


def extract_qdata(html):
    pattern = r'q-data="([^"]*)"'
    matches = re.findall(pattern, html)
    for match in matches:
        try:
            decoded = match.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
            data = json.loads(decoded)
            if "totalStats" in data and "averageStats" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def main():
    all_team_data = {}

    for stat_id, stat_name in STATS.items():
        print(f"Fetching {stat_name} (id={stat_id})...", file=sys.stderr)
        html = fetch_page(stat_id)
        if not html:
            print(f"  FAILED to fetch", file=sys.stderr)
            continue

        if len(html) < 1000:
            print(f"  Page too small ({len(html)} bytes) - likely blocked", file=sys.stderr)
            continue

        data = extract_qdata(html)
        if not data:
            print(f"  Could not find q-data in HTML ({len(html)} bytes)", file=sys.stderr)
            continue

        avg_stats = data["averageStats"]
        total_stats = data["totalStats"]

        for leader in avg_stats["leaders"]:
            team = leader["teamNickName"]
            if team not in all_team_data:
                all_team_data[team] = {"played": leader["played"]}
            all_team_data[team][f"{stat_name}_avg"] = leader["value"]

        for leader in total_stats["leaders"]:
            team = leader["teamNickName"]
            all_team_data[team][f"{stat_name}_total"] = leader["value"]

        print(f"  Got {len(avg_stats['leaders'])} teams", file=sys.stderr)
        time.sleep(1)

    print(f"\n=== Results: {len(all_team_data)} teams ===\n", file=sys.stderr)

    # Output as Python dict
    print("TEAM_STATS = {")
    for team in sorted(all_team_data.keys()):
        d = all_team_data[team]
        print(f'    "{team}": {json.dumps(d)},')
    print("}")


if __name__ == "__main__":
    main()
