"""Item type_id -> integer index mapping.

C# sends item type_id as a string FourCC (e.g. "I001").
We map these to integer indices for observation encoding.
Index 0 = empty slot / unknown item.
"""

# Known item type_ids from the map
# TODO: Extract full list from war3map.w3t
ITEM_IDS = [
    "I001", "I002", "I003", "I004", "I005", "I006", "I007", "I008",
    "I009", "I00A", "I00B", "I00C", "I00D", "I00E", "I00F", "I00G",
]

ITEM_TO_IDX = {item_id: i + 1 for i, item_id in enumerate(ITEM_IDS)}
# 0 = empty/unknown
NUM_ITEMS = len(ITEM_IDS) + 1  # +1 for empty slot

def item_index(type_id: str | None) -> int:
    """Return integer index for an item type_id. 0 = empty/unknown."""
    if type_id is None:
        return 0
    return ITEM_TO_IDX.get(type_id, 0)
