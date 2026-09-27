#!/usr/bin/env python3
"""Self-test for check_security_fix_has_test.py path classification."""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "security_fix_gate", os.path.join(ROOT, "ci", "check_security_fix_has_test.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    gate = _load()
    cases = [
        ("audit/test_exploit_batch_soundness.cpp", True),
        ("tests/ci/test_mutation_reporting.py", True),
        ("compat/libsecp256k1_shim/tests/shim_test.cpp", True),
        ("compat/libbitcoin_bridge/tests/test_lbtc_bridge.cpp", True),
        ("compat/libbitcoin_direct/tests/test_direct_verify.cpp", True),
        ("compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp", False),
        # 2026-09-25: added to cover retroactive entries for security gate script changes themselves
        # (the 8998cd89 coverage + frozen count increments touch the SECURITY_CI_FILE).
        ("docs/AUDIT_CHANGELOG.md", False),
    ]

    failures = []
    for path, expected in cases:
        actual = bool(gate.is_test_file(path))
        if actual != expected:
            failures.append(f"{path}: expected is_test_file={expected}, got {actual}")

    # Every retroactive-coverage entry must name test files that still exist.
    # check_commit() turns a missing one into a hard FAIL on a commit that was
    # already accepted, so a test deleted or renamed later would break the gate
    # on an unrelated push. Catch it here instead, where the fix is obvious.
    for sha, (test_files, _note) in gate.RETROACTIVELY_COVERED.items():
        for test_file in test_files:
            if not os.path.exists(os.path.join(ROOT, test_file)):
                failures.append(
                    f"RETROACTIVELY_COVERED[{sha}] names a missing file: {test_file}"
                )

    if failures:
        print("check_security_fix_has_test self-test: FAILED")
        for failure in failures:
            print("  - " + failure)
        return 1

    print("check_security_fix_has_test self-test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
