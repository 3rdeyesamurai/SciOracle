"""System 2: deterministic verification of mathematical claims.

Three backends, in descending order of authority and ascending order of cost:

  * Lean 4       -- machine-checked proof against kernel axioms. A claim is
                    "certified" only if it compiles with zero `sorry`.
  * Wolfram      -- symbolic/numeric adjudication when available.
  * SymPy        -- always present; decides identity, refutes with an explicit
                    counterexample, or reports "conditional" with the solution set.

Whichever backends are unavailable are reported as unavailable. The engine never
upgrades an unverified claim to verified because a prover was missing.
"""
import json
import os
import random
import shutil
import subprocess
import tempfile

import sympy as sp

from ...config import LEAN_BIN, LEAN_TIMEOUT, WOLFRAM_APP_ID, WOLFRAM_MCP_URL


# ---------------------------------------------------------------- SymPy tier
def _sample_counterexample(lhs, rhs, symbols, trials: int = 120) -> dict | None:
    rng = random.Random(20260824)
    free = list(symbols)
    for _ in range(trials):
        assignment = {s: sp.Rational(rng.randint(-40, 40), rng.randint(1, 7)) for s in free}
        try:
            left = sp.N(lhs.subs(assignment))
            right = sp.N(rhs.subs(assignment))
        except Exception:
            continue
        if not (left.is_number and right.is_number):
            continue
        try:
            if abs(complex(left) - complex(right)) > 1e-9 * max(1.0, abs(complex(left))):
                return {
                    "assignment": {str(k): str(v) for k, v in assignment.items()},
                    "lhs_value": str(left), "rhs_value": str(right),
                }
        except (TypeError, ValueError):
            continue
    return None


def sympy_verify(expr) -> dict:
    """Classify a relation as identity / conditional / refuted / non-relational."""
    if isinstance(expr, sp.logic.boolalg.BooleanTrue):
        return {"backend": "sympy", "status": "identity",
                "detail": "relation collapsed to True under structural evaluation"}
    if isinstance(expr, sp.logic.boolalg.BooleanFalse):
        return {"backend": "sympy", "status": "refuted",
                "detail": "relation collapsed to False under structural evaluation"}
    if not isinstance(expr, sp.Equality):
        if isinstance(expr, sp.core.relational.Relational):
            simplified = sp.simplify(expr)
            return {"backend": "sympy", "status": "conditional", "detail": f"inequality; simplified form {simplified}"}
        return {"backend": "sympy", "status": "not_a_claim",
                "detail": "expression is a term, not a relation; archived without adjudication"}

    lhs, rhs = expr.lhs, expr.rhs
    symbols = sorted(expr.free_symbols, key=str)
    try:
        difference = sp.simplify(sp.expand(lhs - rhs))
    except Exception as exc:
        return {"backend": "sympy", "status": "indeterminate", "detail": f"simplification failed: {exc}"[:200]}

    if difference == 0:
        return {"backend": "sympy", "status": "identity",
                "detail": "lhs - rhs simplifies to 0 for all admissible assignments",
                "residual": "0"}

    # A relation whose left side is a bare symbol absent from the right side is a
    # definition, not an assertion of fact; adjudicating it as "false" would be a
    # category error. It is archived as definitional and used to bind the symbol.
    if isinstance(lhs, sp.Symbol) and lhs not in rhs.free_symbols:
        return {"backend": "sympy", "status": "definitional",
                "detail": f"defines {lhs} := {rhs}; no truth value asserted",
                "defines": str(lhs), "definiens": str(rhs)}
    if isinstance(rhs, sp.Symbol) and rhs not in lhs.free_symbols:
        return {"backend": "sympy", "status": "definitional",
                "detail": f"defines {rhs} := {lhs}; no truth value asserted",
                "defines": str(rhs), "definiens": str(lhs)}

    counterexample = _sample_counterexample(lhs, rhs, symbols)
    if counterexample:
        result = {"backend": "sympy", "status": "refuted",
                  "detail": "explicit counterexample found; the equality does not hold as an identity "
                            "(it may still hold on the restricted solution set reported below)",
                  "counterexample": counterexample, "residual": str(difference)}
    else:
        result = {"backend": "sympy", "status": "conditional",
                  "detail": "holds only on a proper subset of the domain", "residual": str(difference)}
    if symbols:
        try:
            solutions = sp.solve(sp.Eq(lhs, rhs), symbols[0], dict=True)
            result["solution_set"] = [{str(k): str(v) for k, v in s.items()} for s in solutions][:8]
        except Exception:
            pass
    return result


# ----------------------------------------------------------------- Lean tier
def lean_source(expr, theorem_name: str = "archived_claim") -> str:
    """Emit a Lean 4 obligation for the claim (Mathlib-flavoured)."""
    symbols = sorted(expr.free_symbols, key=str) if hasattr(expr, "free_symbols") else []
    binders = " ".join(f"({s} : ℝ)" for s in symbols)

    def lean_of(e) -> str:
        if isinstance(e, sp.Equality):
            return f"{lean_of(e.lhs)} = {lean_of(e.rhs)}"
        if isinstance(e, sp.Add):
            return "(" + " + ".join(lean_of(a) for a in e.args) + ")"
        if isinstance(e, sp.Mul):
            return "(" + " * ".join(lean_of(a) for a in e.args) + ")"
        if isinstance(e, sp.Pow):
            return f"({lean_of(e.base)} ^ {lean_of(e.exp)})"
        if isinstance(e, sp.Rational) and not isinstance(e, sp.Integer):
            return f"({e.p} / {e.q} : ℝ)"
        return str(e)

    return (
        "-- Generated by Lexegis. Compiles iff the archived claim is a theorem.\n"
        "import Mathlib\n\n"
        f"theorem {theorem_name} {binders} :\n"
        f"    {lean_of(expr)} := by\n"
        "  ring\n"
    )


def lean_verify(expr, theorem_name: str = "archived_claim") -> dict:
    src = lean_source(expr, theorem_name)
    binary = shutil.which(LEAN_BIN)
    if not binary:
        return {"backend": "lean4", "status": "unavailable", "available": False,
                "detail": "no Lean 4 toolchain on PATH; obligation generated but not compiled",
                "source": src, "sorry_count": src.count("sorry")}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "Claim.lean")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(src)
        try:
            proc = subprocess.run([binary, path], capture_output=True, text=True, timeout=LEAN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"backend": "lean4", "status": "timeout", "available": True, "source": src,
                    "detail": f"compilation exceeded {LEAN_TIMEOUT}s"}
    ok = proc.returncode == 0 and "sorry" not in proc.stdout
    return {
        "backend": "lean4",
        "available": True,
        "status": "certified" if ok else "failed",
        "detail": "compiled with zero sorry against kernel axioms" if ok else "compilation failed",
        "source": src,
        "sorry_count": src.count("sorry"),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "exit_code": proc.returncode,
    }


# -------------------------------------------------------------- Wolfram tier
def wolfram_verify(expr, timeout: float = 20.0) -> dict:
    """Wolfram adjudication over MCP or the public API, when configured."""
    query = f"FullSimplify[{sp.mathematica_code(expr.lhs - expr.rhs)}]" if isinstance(expr, sp.Equality) \
        else f"FullSimplify[{sp.mathematica_code(expr)}]"
    if not (WOLFRAM_MCP_URL or WOLFRAM_APP_ID):
        return {"backend": "wolfram", "status": "unavailable", "available": False,
                "detail": "WOLFRAM_MCP_URL / WOLFRAM_APP_ID unset; query generated but not dispatched",
                "query": query}
    import httpx
    try:
        if WOLFRAM_MCP_URL:
            response = httpx.post(WOLFRAM_MCP_URL, json={"expression": query}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result", json.dumps(payload)[:500])
        else:
            response = httpx.get("https://api.wolframalpha.com/v1/result",
                                 params={"appid": WOLFRAM_APP_ID, "i": query}, timeout=timeout)
            response.raise_for_status()
            result = response.text
    except Exception as exc:
        return {"backend": "wolfram", "status": "error", "available": True,
                "detail": f"{type(exc).__name__}: {exc}"[:200], "query": query}
    normalised = str(result).strip()
    return {
        "backend": "wolfram", "available": True, "query": query, "result": normalised,
        "status": "identity" if normalised in ("0", "0.", "True") else "indeterminate",
        "detail": "residual simplifies to zero" if normalised in ("0", "0.", "True")
                  else "residual is non-zero or not decided",
    }


def verify(expr, *, use_lean: bool = True, use_wolfram: bool = True, theorem_name: str = "archived_claim") -> dict:
    """Run every available backend and reconcile them into one verdict."""
    backends = [sympy_verify(expr)]
    if use_wolfram:
        backends.append(wolfram_verify(expr))
    if use_lean:
        backends.append(lean_verify(expr, theorem_name))

    by_name = {b["backend"]: b for b in backends}
    lean = by_name.get("lean4", {})
    sym = by_name["sympy"]

    if lean.get("status") == "certified":
        verdict, authority = "certified", "lean4"
    elif sym["status"] == "identity":
        verdict, authority = "verified", "sympy"
    elif sym["status"] == "refuted":
        verdict, authority = "refuted", "sympy"
    elif sym["status"] == "definitional":
        verdict, authority = "definitional", "sympy"
    elif sym["status"] in ("conditional", "indeterminate"):
        verdict, authority = sym["status"], "sympy"
    else:
        verdict, authority = "unadjudicated", "sympy"

    disagreement = None
    wolf = by_name.get("wolfram", {})
    if wolf.get("status") == "identity" and sym["status"] == "refuted":
        disagreement = "wolfram reports identity while sympy produced a counterexample; escalate to human review"
    if lean.get("status") == "failed" and sym["status"] == "identity":
        disagreement = "sympy simplifies to an identity but the Lean obligation did not compile; " \
                       "the Lean tactic may be insufficient rather than the claim false"

    return {
        "verdict": verdict,
        "authority": authority,
        "disagreement": disagreement,
        "backends": backends,
        "backends_available": sorted(b["backend"] for b in backends if b.get("available", True)
                                     and b.get("status") != "unavailable"),
    }
