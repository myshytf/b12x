"""Copy-only query-gather routing for a five/four PCIe switch split."""

from __future__ import annotations


def normalize_switch_groups(groups, world_size: int):
    if not groups:
        return ()
    if isinstance(groups, str):
        groups = tuple(tuple(int(x) for x in g.split(",")) for g in groups.split(";"))
    groups = tuple(tuple(int(x) for x in g) for g in groups)
    if (
        world_size != 9
        or len(groups) != 2
        or sorted(map(len, groups)) != [4, 5]
        or sorted(x for g in groups for x in g) != list(range(world_size))
    ):
        raise ValueError(
            "query gather switch groups must partition nine ranks into five and four"
        )
    return groups


def packet_routes(groups, source: int, packet: int):
    """Return the two stages of a 16-byte packet's multicast tree.

    Each source sends one copy across the inter-switch link. Relay ownership
    uses upper packet-index bits; lower bits redirect a small fraction of
    local five-rank traffic through the four-rank switch. Separating those
    bits distributes the redirected traffic among relays. This balances GPU
    send volumes to within 1% of the eight-copy all-gather port lower bound.
    """
    mine = next(g for g in groups if source in g)
    other = next(g for g in groups if source not in g)
    peers = tuple(x for x in mine if x != source)
    relay = other[(packet >> 4) % len(other)]
    loopback = peers[packet & 15] if len(mine) == 5 and (packet & 15) < 4 else None
    first = [(source, peer) for peer in peers if peer != loopback]
    first.append((source, relay))
    second = [(relay, peer) for peer in other if peer != relay]
    if loopback is not None:
        second.append((relay, loopback))
    return tuple(first), tuple(second)
