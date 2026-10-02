"""Canonical phenotypic trait names shared by Web and CLI outputs."""

from collections import namedtuple


TraitDefinition = namedtuple("TraitDefinition", "symbol name unit")


CANONICAL_TRAITS = (
    TraitDefinition("FL", "Fruit Length", "mm"),
    TraitDefinition("FW_mean", "Mean Fruit Width", "mm"),
    TraitDefinition("FW_min", "Minimum Fruit Width", "mm"),
    TraitDefinition("FW_max", "Maximum Fruit Width", "mm"),
    TraitDefinition("FA", "Fruit Area", "mm^2"),
    TraitDefinition("MA", "Mesocarp Area", "mm^2"),
    TraitDefinition("MT_mean", "Mean Mesocarp Thickness", "mm"),
    TraitDefinition("MT_min", "Minimum Mesocarp Thickness", "mm"),
    TraitDefinition("MT_max", "Maximum Mesocarp Thickness", "mm"),
    TraitDefinition("PA", "Placenta Area", "mm^2"),
    TraitDefinition("SA", "Total Seed Area", "mm^2"),
    TraitDefinition("SC", "Seed Count", "count"),
    TraitDefinition("MAF", "Mesocarp Area Fraction", "dimensionless"),
    TraitDefinition("SCR", "Seed-to-Cavity Area Ratio", "dimensionless"),
    TraitDefinition("FCI", "Fruit Curvature Index", "dimensionless"),
    TraitDefinition("FAI", "Fruit Asymmetry Index", "dimensionless"),
    TraitDefinition("MWI", "Mesocarp Wrinkle Index", "dimensionless"),
)


def trait_column(definition):
    """Return the stable human-readable CSV/Web column label."""
    return f"{definition.name} ({definition.symbol})"


CANONICAL_TRAIT_COLUMNS = tuple(trait_column(item) for item in CANONICAL_TRAITS)


def trait_values_by_symbol(chili):
    """Extract all canonical traits without changing legacy Chili attributes."""
    widths = chili.width_fruit_stats
    thickness = chili.mesocarp_thickness_stats
    seeds = chili.seed_stats
    return {
        "FL": chili.length_fruit,
        "FW_mean": chili.width_fruit,
        "FW_min": widths["min"],
        "FW_max": widths["max"],
        "FA": chili.area_fruit,
        "MA": chili.mesocarp_area,
        "MT_mean": thickness["avg_mesocarp_thickness"],
        "MT_min": thickness["min_mesocarp_thickness"],
        "MT_max": thickness["max_mesocarp_thickness"],
        "PA": chili.placenta_area,
        "SA": seeds["sum"],
        "SC": chili.number_seeds,
        "MAF": chili.mesocarp_area_fraction,
        "SCR": chili.seed_density,
        "FCI": chili.curvature_index,
        "FAI": chili.asymmetry_index,
        "MWI": chili.mesocarp_wrinkle_index,
    }


def canonical_trait_row(chili):
    """Return canonical values keyed by the public CSV/Web labels."""
    values = trait_values_by_symbol(chili)
    return {
        trait_column(definition): values[definition.symbol]
        for definition in CANONICAL_TRAITS
    }
