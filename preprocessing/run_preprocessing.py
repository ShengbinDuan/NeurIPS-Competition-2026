"""Run the original workflow from raw data, or resume an explicit cache stage."""

import argparse
import os
from pathlib import Path


STEPS = (
    "is_load_cached_mat", "is_build_valid_raw", "is_view_waveforms",
    "is_exclude_bads_and_interpolate", "is_pre_filter", "is_set_reference",
    "is_ica_cache", "is_gen_components_topomap", "is_gen_properties",
    "is_mark_artifacts_and_build", "is_after_filter",
)


def workflow(stage):
    settings = dict.fromkeys(STEPS, False)
    settings.update(is_load_cached_mat=True, is_build_valid_raw=True)
    if stage == "waveforms":
        settings["is_view_waveforms"] = True
    else:
        settings.update(
            is_exclude_bads_and_interpolate=True, is_pre_filter=True,
            is_set_reference=True, is_ica_cache=True,
        )
        if stage == "ica":
            settings["is_gen_components_topomap"] = True
        elif stage == "properties":
            settings["is_gen_properties"] = True
        elif stage == "clean":
            settings.update(is_mark_artifacts_and_build=True, is_after_filter=True)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("waveforms", "ica", "properties", "clean"), default="ica")
    parser.add_argument("--start", choices=STEPS, default="is_load_cached_mat",
                        help="Explicit cache restore point; existing cache is required for intermediate steps.")
    args = parser.parse_args()
    settings = workflow(args.stage)
    if not settings[args.start]:
        parser.error(f"{args.start} is not enabled for stage {args.stage}")

    project_root = Path(__file__).resolve().parent
    previous_directory = Path.cwd()
    try:
        os.chdir(project_root)
        from src.Preprocessing import main as run

        return run(is_dic=settings, start=args.start)
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
