from pathlib import Path

import ezkl

from zkllms import backend


def verify_proof(
    proof_path: Path,
    vk_path: Path,
    settings_path: Path,
    srs_path: Path,
) -> bool:
    return backend.run(
        ezkl.verify, str(proof_path), str(settings_path), str(vk_path), str(srs_path)
    )
