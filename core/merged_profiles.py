# ============================================================
# G7FX Signal Engine — Merged Profile Detector (Gap 2 Fix)
# ============================================================
# What NV actually does (corrected understanding):
#
#   When two bell curves sit adjacent to each other with a
#   thin volume gap (LVN) between them, price travels through
#   that gap FAST because no one defended those prices before.
#
#   The TARGET is the far boundary of the adjacent profile —
#   not VWAP, not a fixed pip count. It's where the next
#   cluster of institutional interest begins.
#
#   This module:
#   1. Detects when two profiles overlap or sit adjacent
#   2. Identifies the LVN corridor between them
#   3. Sets the dynamic target as the adjacent profile's
#      VAL (for longs) or VAH (for shorts)
# ============================================================

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from core.stage1_amt import VolumeProfile

logger = logging.getLogger(__name__)

PIP = 0.01   # 1 pip for JPY pairs


@dataclass
class MergedProfileResult:
    has_adjacent_profile: bool
    lvn_exists: bool
    lvn_top: float
    lvn_bottom: float
    lvn_width_pips: float
    adjacent_profile_target: Optional[float]   # The VAL or VAH of the next profile
    adjacent_poc: Optional[float]
    composite_poc: Optional[float]             # POC of merged profiles combined
    travel_direction: str                      # "up" | "down" | "none"
    notes: list = field(default_factory=list)


def detect_lvn(profile: VolumeProfile,
               pip_threshold: float = 5.0) -> tuple:
    """
    Scan a profile's volume distribution for Low Volume Nodes —
    price levels where volume is less than 20% of the POC volume.

    Returns list of (low, high) tuples representing LVN zones.
    """
    if not profile.distribution:
        return []

    poc_volume  = profile.distribution.get(profile.poc, 1)
    lvn_threshold = poc_volume * 0.20   # less than 20% of POC = LVN

    lvn_levels  = sorted([
        price for price, vol in profile.distribution.items()
        if vol < lvn_threshold
    ])

    if not lvn_levels:
        return []

    # Group consecutive LVN prices into zones
    zones = []
    zone_start = lvn_levels[0]
    zone_end   = lvn_levels[0]

    for lvl in lvn_levels[1:]:
        if lvl - zone_end <= pip_threshold * PIP * 2:
            zone_end = lvl
        else:
            if (zone_end - zone_start) / PIP >= pip_threshold:
                zones.append((zone_start, zone_end))
            zone_start = lvl
            zone_end   = lvl

    if (zone_end - zone_start) / PIP >= pip_threshold:
        zones.append((zone_start, zone_end))

    return zones


def profiles_are_adjacent(profile_a: VolumeProfile,
                           profile_b: VolumeProfile,
                           gap_pip_threshold: float = 30.0) -> bool:
    """
    Check if two profiles sit next to each other (adjacent but not
    overlapping) with a gap small enough to be an LVN corridor.

    Adjacent = the bottom of B is within gap_pip_threshold pips
               of the top of A (or vice versa), but they don't
               fully overlap.
    """
    gap_threshold = gap_pip_threshold * PIP

    # Profile A is the lower one
    if profile_a.vah < profile_b.val:
        gap = profile_b.val - profile_a.vah
        return gap <= gap_threshold

    # Profile B is the lower one
    if profile_b.vah < profile_a.val:
        gap = profile_a.val - profile_b.vah
        return gap <= gap_threshold

    return False   # they overlap — not adjacent


def profiles_overlap(profile_a: VolumeProfile,
                     profile_b: VolumeProfile) -> bool:
    """Check if two value areas overlap."""
    return not (profile_a.vah < profile_b.val or
                profile_b.vah < profile_a.val)


def analyse_merged_profiles(current_profile: VolumeProfile,
                             prev_profiles: List[VolumeProfile],
                             current_price: float,
                             direction: str) -> MergedProfileResult:
    """
    Main function: given current and previous session profiles,
    detect whether NV's merged-profile targeting applies.

    direction: "long" | "short"

    Logic:
    - If prev profile is ADJACENT (small gap = LVN between them):
        → LVN is the fast-travel corridor
        → Target = far boundary of the adjacent profile
    - If prev profile OVERLAPS:
        → Profiles are merged into one composite
        → POC of composite is the key level
        → Target is still the composite VAH or VAL
    """
    result = MergedProfileResult(
        has_adjacent_profile    = False,
        lvn_exists              = False,
        lvn_top                 = 0.0,
        lvn_bottom              = 0.0,
        lvn_width_pips          = 0.0,
        adjacent_profile_target = None,
        adjacent_poc            = None,
        composite_poc           = None,
        travel_direction        = "none",
        notes                   = []
    )

    if not prev_profiles:
        result.notes.append("No previous profiles — standard VWAP targeting applies")
        return result

    prev = prev_profiles[-1]   # most recent prior session

    # ── Case 1: Profiles are adjacent (LVN corridor between them) ──
    if profiles_are_adjacent(current_profile, prev):
        result.has_adjacent_profile = True

        # Identify which profile is above/below
        if prev.val > current_profile.vah:
            # prev profile is ABOVE current — relevant for longs
            lvn_bottom = current_profile.vah
            lvn_top    = prev.val
            travel_dir = "up"
            target     = prev.vah   # far boundary of upper profile
            adj_poc    = prev.poc

        elif prev.vah < current_profile.val:
            # prev profile is BELOW current — relevant for shorts
            lvn_bottom = prev.vah
            lvn_top    = current_profile.val
            travel_dir = "down"
            target     = prev.val   # far boundary of lower profile
            adj_poc    = prev.poc

        else:
            result.notes.append("Adjacent check inconclusive")
            return result

        lvn_width = (lvn_top - lvn_bottom) / PIP

        result.lvn_exists              = True
        result.lvn_top                 = round(lvn_top, 3)
        result.lvn_bottom              = round(lvn_bottom, 3)
        result.lvn_width_pips          = round(lvn_width, 1)
        result.adjacent_profile_target = round(target, 3)
        result.adjacent_poc            = round(adj_poc, 3)
        result.travel_direction        = travel_dir
        result.notes.append(
            f"Adjacent profiles detected — LVN corridor "
            f"{lvn_bottom:.3f}–{lvn_top:.3f} ({lvn_width:.0f} pips)"
        )
        result.notes.append(
            f"Price travels {travel_dir} through LVN — "
            f"target = adjacent profile boundary {target:.3f}"
        )
        logger.info(
            f"MergedProfile | Adjacent | LVN={lvn_bottom:.3f}-{lvn_top:.3f} "
            f"| target={target:.3f} | dir={travel_dir}"
        )

    # ── Case 2: Profiles overlap — build composite ──
    elif profiles_overlap(current_profile, prev):
        result.has_adjacent_profile = True

        # Composite value area = union of both
        composite_val = min(current_profile.val, prev.val)
        composite_vah = max(current_profile.vah, prev.vah)

        # Composite POC = whichever single POC has higher volume
        curr_poc_vol = current_profile.distribution.get(current_profile.poc, 0)
        prev_poc_vol = prev.distribution.get(prev.poc, 0)

        composite_poc = (current_profile.poc if curr_poc_vol >= prev_poc_vol
                         else prev.poc)

        result.composite_poc = round(composite_poc, 3)

        # LVN within the composite — scan between the two POCs
        if current_profile.poc < prev.poc:
            lvn_bottom = current_profile.poc
            lvn_top    = prev.poc
        else:
            lvn_bottom = prev.poc
            lvn_top    = current_profile.poc

        lvn_width = (lvn_top - lvn_bottom) / PIP
        result.lvn_exists     = lvn_width > 5
        result.lvn_bottom     = round(lvn_bottom, 3)
        result.lvn_top        = round(lvn_top, 3)
        result.lvn_width_pips = round(lvn_width, 1)

        # Target is the far boundary of the composite
        if direction == "long":
            result.adjacent_profile_target = round(composite_vah, 3)
            result.travel_direction        = "up"
        else:
            result.adjacent_profile_target = round(composite_val, 3)
            result.travel_direction        = "down"

        result.notes.append(
            f"Overlapping profiles — composite VAL={composite_val:.3f} "
            f"VAH={composite_vah:.3f} POC={composite_poc:.3f}"
        )
        result.notes.append(
            f"Target = composite boundary {result.adjacent_profile_target:.3f}"
        )
        logger.info(
            f"MergedProfile | Overlapping | POC={composite_poc:.3f} "
            f"| target={result.adjacent_profile_target:.3f}"
        )

    else:
        result.notes.append(
            "Profiles not adjacent/overlapping — "
            "standard profile targeting applies"
        )

    return result
