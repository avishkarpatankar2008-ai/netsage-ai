"""
Starter deterministic checks for NetSage Lite.

This is intentionally a small set to get the workspace UI working end to
end -- three checks, each looking for one specific, unambiguous problem in
pasted command output. Add more functions and register them in RULES below
as you cover more fault types; nothing else in the app needs to change when
you do.

Each check function takes the list of {"source", "text"} command-output
blocks and returns a dict {"rule_id", "status", "finding"} or None if the
case doesn't contain the input that check needs.
"""

from __future__ import annotations

import ipaddress
import re

IPV4_PATTERN = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def _blocks_matching(command_output: list[dict], keyword: str) -> list[dict]:
    keyword = keyword.lower()
    return [b for b in command_output if keyword in b["source"].lower()]


def _extract_ipconfig_fields(text: str) -> dict:
    """Pull IP / mask / gateway out of a Packet Tracer `ipconfig` block."""
    fields = {"ip": "", "mask": "", "gateway": ""}
    for line in text.splitlines():
        line = line.strip()
        match = IPV4_PATTERN.search(line)
        if not match:
            continue
        if line.lower().startswith("ip address"):
            fields["ip"] = match.group(0)
        elif line.lower().startswith("subnet mask"):
            fields["mask"] = match.group(0)
        elif line.lower().startswith("default gateway"):
            fields["gateway"] = match.group(0)
    return fields


def check_duplicate_ip(command_output: list[dict]) -> dict | None:
    ip_owners: dict[str, list[str]] = {}
    for block in _blocks_matching(command_output, "ipconfig"):
        fields = _extract_ipconfig_fields(block["text"])
        if fields["ip"]:
            ip_owners.setdefault(fields["ip"], []).append(block["source"])

    if len(ip_owners) < 2:
        return None  # not enough data to say anything meaningful

    duplicates = {ip: owners for ip, owners in ip_owners.items() if len(owners) > 1}
    if duplicates:
        detail = "; ".join(f"{ip} claimed by {', '.join(owners)}" for ip, owners in duplicates.items())
        return {"rule_id": "DUPLICATE-IP", "status": "fail", "finding": f"Same address used twice: {detail}"}
    return {"rule_id": "DUPLICATE-IP", "status": "pass", "finding": "No repeated IPv4 addresses found."}


def check_gateway_same_subnet(command_output: list[dict]) -> dict | None:
    problems = []
    checked = 0
    for block in _blocks_matching(command_output, "ipconfig"):
        fields = _extract_ipconfig_fields(block["text"])
        if not (fields["ip"] and fields["mask"] and fields["gateway"]):
            continue
        checked += 1
        try:
            host_net = ipaddress.ip_interface(f"{fields['ip']}/{fields['mask']}").network
            gateway_addr = ipaddress.ip_address(fields["gateway"])
        except ValueError:
            problems.append(f"{block['source']}: could not parse IP/mask/gateway as valid IPv4 values.")
            continue
        if gateway_addr not in host_net:
            problems.append(
                f"{block['source']}: gateway {fields['gateway']} is not inside {host_net}, "
                f"so the host ({fields['ip']}/{fields['mask']}) can't reach it directly."
            )

    if checked == 0:
        return None
    if problems:
        return {"rule_id": "GATEWAY-SUBNET", "status": "fail", "finding": " ".join(problems)}
    return {"rule_id": "GATEWAY-SUBNET", "status": "pass", "finding": f"Gateway is reachable on-subnet for {checked} host(s)."}


def check_interface_down(command_output: list[dict]) -> dict | None:
    down_lines = []
    checked_any = False
    for block in _blocks_matching(command_output, "interface"):
        for line in block["text"].splitlines():
            lowered = line.lower()
            if "administratively down" in lowered or re.search(r"\bdown\s+down\b", lowered):
                checked_any = True
                down_lines.append(f"{block['source']}: {line.strip()}")
            elif re.search(r"\bup\s+up\b", lowered):
                checked_any = True

    if not checked_any:
        return None
    if down_lines:
        return {"rule_id": "IF-DOWN", "status": "fail", "finding": "Interface(s) not up/up: " + "; ".join(down_lines)}
    return {"rule_id": "IF-DOWN", "status": "pass", "finding": "All referenced interfaces show up/up."}


# Register every check here -- the UI just loops over this list, so adding
# a new function above and listing it here is the only step needed.
RULES = [check_duplicate_ip, check_gateway_same_subnet, check_interface_down]


def run_all(command_output: list[dict]) -> list[dict]:
    results = []
    for rule_fn in RULES:
        try:
            outcome = rule_fn(command_output)
        except Exception as exc:  # a broken parser shouldn't kill the other checks
            outcome = {"rule_id": rule_fn.__name__, "status": "fail", "finding": f"Check crashed: {exc}"}
        if outcome is not None:
            results.append(outcome)
    return results
