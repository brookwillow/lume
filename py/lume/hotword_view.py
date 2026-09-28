"""Inspect the persisted personalized ASR hotword pool."""

import argparse
import json

from .asr_lexicon import ASRLexiconEngine, ASRLexiconSettings
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Lume personalized ASR hotwords")
    parser.add_argument("--scene", default="", help="optional scene used for ranking, e.g. navigation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--refresh", action="store_true", help="refresh the local application entity index before listing")
    args = parser.parse_args()

    config = load_config()
    engine = ASRLexiconEngine(ASRLexiconSettings.from_config(config.asr))
    if args.refresh:
        path = engine.refresh_local_entities(reason="manual_view")
        if not args.json:
            print(f"Refreshed local entity index: {path}")
    hotwords = engine.recall_hotwords(scene=args.scene, limit=args.limit)
    if args.json:
        print(json.dumps(hotwords, ensure_ascii=False, indent=2))
        return
    if not hotwords:
        print("No active personalized hotwords.")
        return
    print(f"Personalized hotwords (scene={args.scene or 'any'}):")
    for index, item in enumerate(hotwords, start=1):
        print(f"{index:>2}. {item['text']}  weight={item['weight']:.3f}  type={item['entity_type']}  source={item.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
