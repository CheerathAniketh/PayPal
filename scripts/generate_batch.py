"""Build and freeze the synthetic batch.

Run:  python -m scripts.generate_batch
"""

from __future__ import annotations

from config.taxonomy import FailureClass
from recoup.generator import freeze, generate
from recoup.money import format_inr


def main() -> None:
    batch = generate()
    paths = freeze(batch)

    dead = [rid for rid, t in batch.latent.items() if t.is_truly_dead]
    class_five_dead = sum(
        1 for rid in dead
        if batch.latent[rid].seeded_class is FailureClass.DO_NOT_HONOUR
    )

    print("Froze synthetic batch:")
    for name, path in paths.items():
        print(f"  {name:<7} -> {path}")
    print()
    print(
        f"{len(batch.records)} records | {len(batch.customers)} customers "
        f"({batch.recurring_customer_count()} recurring) | "
        f"{format_inr(batch.total_at_risk_cents())} at risk"
    )
    print(
        f"{len(dead)} seeded-dead ({class_five_dead} of them class 5) | "
        f"holdout {len(batch.holdout_records)} records / "
        f"{len(batch.holdout_customer_ids)} disjoint customers"
    )
    print(f"seed {batch.seed}")


if __name__ == "__main__":
    main()
