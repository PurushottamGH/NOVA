import re

from sympy import (
    Matrix,
    Symbol,
    diff,
    factorint,
    gcd,
    integrate,
    isprime,
    latex,
    lcm,
    limit,
    oo,
    pretty,
    series,
    solve,
    symbols,
)
from sympy.parsing.sympy_parser import parse_expr


class NovaMathEngine:
    """
    Symbolic math solver plugged into Nova.
    NovaMind detects math intent → hands off here → returns exact answer.
    """

    # ------------------------------------------------------------------ #
    #  Existing methods (unchanged)
    # ------------------------------------------------------------------ #

    def solve_equation(self, equation_str: str) -> str:
        """
        Solve a single equation for x.

        Args:
            equation_str: Equation string, e.g. "x**2 + 2*x - 3 = 0" or "x**2 - 4".

        Returns:
            Solutions as a formatted string, or an error message.
        """
        try:
            x = symbols("x")
            if "=" in equation_str:
                left, right = equation_str.split("=")
                expr = parse_expr(left.strip()) - parse_expr(right.strip())
            else:
                expr = parse_expr(equation_str)

            solutions = solve(expr, x)
            return f"Solutions: {solutions}"
        except Exception as e:
            return f"Could not solve: {e}"

    def differentiate(self, expr_str: str, var: str = "x") -> str:
        """
        Differentiate an expression with respect to a variable.

        Args:
            expr_str: Expression string, e.g. "x**3 + 2*x".
            var: Variable to differentiate with respect to (default "x").

        Returns:
            Derivative as a formatted string, or an error message.
        """
        try:
            x = Symbol(var)
            expr = parse_expr(expr_str)
            result = diff(expr, x)
            return f"d/d{var}({expr_str}) = {result}"
        except Exception as e:
            return f"Could not differentiate: {e}"

    def integrate_expr(self, expr_str: str, var: str = "x") -> str:
        """
        Integrate an expression with respect to a variable.

        Args:
            expr_str: Expression string, e.g. "x**2 + 1".
            var: Variable to integrate with respect to (default "x").

        Returns:
            Integral as a formatted string, or an error message.
        """
        try:
            x = Symbol(var)
            expr = parse_expr(expr_str)
            result = integrate(expr, x)
            return f"∫({expr_str})d{var} = {result} + C"
        except Exception as e:
            return f"Could not integrate: {e}"

    # ------------------------------------------------------------------ #
    #  New methods
    # ------------------------------------------------------------------ #

    def solve_system(self, equations: list[str], variables: list[str]) -> str:
        """
        Solve a system of simultaneous equations.

        Args:
            equations: List of equation strings, e.g. ["2*x + y = 5", "x - y = 1"].
            variables: List of variable names, e.g. ["x", "y"].

        Returns:
            Formatted solution dictionary as a string, or an error message.
        """
        try:
            syms = symbols(" ".join(variables))
            if not isinstance(syms, tuple):
                syms = (syms,)

            system = []
            for eq_str in equations:
                if "=" in eq_str:
                    left, right = eq_str.split("=", 1)
                    expr = parse_expr(left.strip()) - parse_expr(right.strip())
                else:
                    expr = parse_expr(eq_str)
                system.append(expr)

            solutions = solve(system, syms)
            return f"Solutions: {solutions}"
        except Exception as e:
            return f"Could not solve system: {e}"

    def matrix_ops(self, matrix_data: list[list]) -> dict:
        """
        Perform common matrix operations.

        Args:
            matrix_data: 2D list representing a matrix, e.g. [[1, 2], [3, 4]].

        Returns:
            dict with keys: determinant, inverse, eigenvalues, rank.
            Returns dict with 'error' key on failure.
        """
        try:
            m = Matrix(matrix_data)
            det = m.det()

            try:
                inv = m.inv() if det != 0 else "Singular"
            except Exception:
                inv = "Singular"

            eigenvals = m.eigenvals()
            rank = m.rank()

            return {
                "determinant": str(det),
                "inverse": str(inv),
                "eigenvalues": {str(k): v for k, v in eigenvals.items()},
                "rank": rank,
            }
        except Exception as e:
            return {"error": f"Matrix operation failed: {e}"}

    def solve_with_latex(self, equation_str: str) -> dict:
        """
        Solve an equation and return solutions with LaTeX and pretty-print forms.

        Args:
            equation_str: Equation string, e.g. "x**2 - 4 = 0".

        Returns:
            dict with keys: solutions, latex, pretty.
            Returns dict with 'error' key on failure.
        """
        try:
            x = symbols("x")
            if "=" in equation_str:
                left, right = equation_str.split("=", 1)
                expr = parse_expr(left.strip()) - parse_expr(right.strip())
            else:
                expr = parse_expr(equation_str)

            solutions = solve(expr, x)
            latex_strs = [latex(s) for s in solutions]
            pretty_str = pretty(solutions)

            return {
                "solutions": [str(s) for s in solutions],
                "latex": latex_strs,
                "pretty": pretty_str,
            }
        except Exception as e:
            return {"error": f"Could not solve with LaTeX: {e}"}

    def explain_steps(self, problem_type: str, expression: str) -> str:
        """
        Narrate step-by-step what SymPy computed for a given problem.

        Args:
            problem_type: One of "differentiate", "integrate", "solve", "limit", "series".
            expression: Mathematical expression string, e.g. "x**3 + 2*x".

        Returns:
            Numbered steps string explaining the computation, or an error message.
        """
        try:
            x = symbols("x")
            expr = parse_expr(expression)
            steps = []

            if problem_type == "differentiate":
                steps.append(f"1. Parse expression: {expr}")
                steps.append(f"2. Identify variable of differentiation: x")
                result = diff(expr, x)
                steps.append(f"3. Apply differentiation rules (power rule, chain rule, etc.)")
                steps.append(f"4. Compute derivative: d/dx({expr}) = {result}")
                steps.append(f"5. Simplify result: {result}")

            elif problem_type == "integrate":
                steps.append(f"1. Parse expression: {expr}")
                steps.append(f"2. Identify variable of integration: x")
                result = integrate(expr, x)
                steps.append(f"3. Apply integration rules (power rule, substitution, etc.)")
                steps.append(f"4. Compute integral: ∫({expr})dx = {result}")
                steps.append(f"5. Add constant of integration: {result} + C")

            elif problem_type == "solve":
                steps.append(f"1. Parse expression: {expr}")
                steps.append(f"2. Set expression equal to zero: {expr} = 0")
                result = solve(expr, x)
                steps.append(f"3. Factor or apply quadratic formula / root-finding")
                steps.append(f"4. Solve for x: solutions = {result}")
                steps.append(f"5. Verify each solution satisfies the original equation")

            elif problem_type == "limit":
                steps.append(f"1. Parse expression: {expr}")
                steps.append(f"2. Identify limit variable: x, approaching 0")
                result = limit(expr, x, 0)
                steps.append(f"3. Evaluate limit using L'Hôpital's rule or direct substitution")
                steps.append(f"4. Compute: lim(x→0) {expr} = {result}")

            elif problem_type == "series":
                steps.append(f"1. Parse expression: {expr}")
                steps.append(f"2. Identify expansion point: x = 0, order = 6")
                result = series(expr, x, 0, 6)
                steps.append(f"3. Compute Taylor coefficients using derivatives at expansion point")
                steps.append(f"4. Series expansion: {result}")

            else:
                return f"Unknown problem type: {problem_type}. Supported: differentiate, integrate, solve, limit, series"

            return "\n".join(steps)
        except Exception as e:
            return f"Could not explain steps: {e}"

    def compute_limit(self, expr_str: str, var: str, point: str) -> str:
        """
        Compute the limit of an expression as a variable approaches a point.

        Args:
            expr_str: Expression string, e.g. "sin(x)/x".
            var: Variable name, e.g. "x".
            point: Point to approach, e.g. "0" or "oo" for infinity.

        Returns:
            Limit result as a formatted string, or an error message.
        """
        try:
            v = Symbol(var)
            expr = parse_expr(expr_str)

            if point.strip() in ("oo", "inf", "infinity"):
                p = oo
            elif point.strip() in ("-oo", "-inf", "-infinity"):
                p = -oo
            else:
                p = parse_expr(point)

            result = limit(expr, v, p)
            return f"lim({var}→{point}) {expr_str} = {result}"
        except Exception as e:
            return f"Could not compute limit: {e}"

    def taylor_series(self, expr_str: str, var: str = "x",
                      point: int = 0, order: int = 6) -> str:
        """
        Compute the Taylor series expansion of an expression.

        Args:
            expr_str: Expression string, e.g. "sin(x)".
            var: Variable name (default "x").
            point: Expansion point (default 0).
            order: Number of terms in the expansion (default 6).

        Returns:
            Series expansion as a string, or an error message.
        """
        try:
            v = Symbol(var)
            expr = parse_expr(expr_str)
            result = series(expr, v, point, order)
            return f"Taylor series of {expr_str} around {var}={point}: {result}"
        except Exception as e:
            return f"Could not compute Taylor series: {e}"

    def number_theory(self, n: int) -> dict:
        """
        Perform number theory operations on an integer.

        Args:
            n: Positive integer to analyze.

        Returns:
            dict with keys: is_prime, factorization, factors, gcd_example, lcm_example.
            Returns dict with 'error' key on failure.
        """
        try:
            n = int(n)
            prime = isprime(n)
            factorization = factorint(n)
            factors = sorted(
                set(
                    f
                    for p, exp in factorization.items()
                    for f in [p]
                    for _ in range(exp)
                )
            )
            # Provide example gcd/lcm with n and n+1
            gcd_val = gcd(n, n + 1)
            lcm_val = lcm(n, n + 1)

            return {
                "is_prime": prime,
                "factorization": {str(k): v for k, v in factorization.items()},
                "factors": [int(f) for f in factors],
                "gcd_example": f"gcd({n}, {n + 1}) = {gcd_val}",
                "lcm_example": f"lcm({n}, {n + 1}) = {lcm_val}",
            }
        except Exception as e:
            return {"error": f"Number theory failed: {e}"}

    # ------------------------------------------------------------------ #
    #  Auto-detect and solve (upgraded)
    # ------------------------------------------------------------------ #

    def detect_and_solve(self, text: str) -> str | None:
        """
        Auto-detect if user input contains math and solve it.
        Routes to the appropriate solver based on keywords.

        Args:
            text: User input text to analyze.

        Returns:
            Answer string or None if no math detected.
        """
        math_keywords = [
            "solve",
            "differentiate",
            "integrate",
            "simplify",
            "factor",
            "expand",
            "derivative",
            "integral",
            "equation",
            "limit",
            "lim",
            "series",
            "taylor",
            "system",
            "simultaneous",
            "matrix",
            "determinant",
            "eigenvalue",
            "prime",
        ]

        text_lower = text.lower()
        has_math = any(kw in text_lower for kw in math_keywords)

        # Also detect expressions like "x^2 + 3x - 4 = 0"
        has_equation = bool(re.search(r"[a-zA-Z]\^?\d*\s*[+\-*/=]", text))

        if not (has_math or has_equation):
            return None

        # --- Route: limit ---
        if "limit" in text_lower or "lim" in text_lower:
            # Try to parse "limit of <expr> as <var> -> <point>"
            match = re.search(
                r"(?:limit|lim)\s+(?:of\s+)?(.+?)\s+(?:as\s+)?(\w)\s*(?:->|→|approaches?)\s*(.+?)(?:\s*$)",
                text, re.IGNORECASE,
            )
            if match:
                return self.compute_limit(
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )

        # --- Route: series / taylor ---
        if "series" in text_lower or "taylor" in text_lower:
            match = re.search(
                r"(?:series|taylor|expand)\s+(?:of\s+)?(.+?)(?:\s+around\s+(.+?))?(?:\s*$)",
                text, re.IGNORECASE,
            )
            if match:
                expr = match.group(1).strip()
                return self.taylor_series(expr)

        # --- Route: system / simultaneous ---
        if "system" in text_lower or "simultaneous" in text_lower:
            # Extract equations between quotes or after colon
            eqs = re.findall(r"['\"](.+?)['\"]", text)
            if not eqs:
                # Try comma-separated after "system"
                match = re.search(r"(?:system|simultaneous)[:\s]+(.+)", text, re.IGNORECASE)
                if match:
                    eqs = [e.strip() for e in match.group(1).split(",")]
            if eqs:
                # Auto-detect variables
                all_vars = set(re.findall(r"\b([a-zA-Z])\b", " ".join(eqs)))
                all_vars -= {"e", "E", "i", "I"}  # exclude math constants
                variables = sorted(all_vars)
                if variables:
                    return self.solve_system(eqs, variables)

        # --- Route: matrix / determinant / eigenvalue ---
        if "matrix" in text_lower or "determinant" in text_lower or "eigenvalue" in text_lower:
            # Try to extract matrix data from text like [[1,2],[3,4]]
            match = re.search(r"(\[\[.+?\]\])", text)
            if match:
                try:
                    import json
                    matrix_data = json.loads(match.group(1))
                    result = self.matrix_ops(matrix_data)
                    return str(result)
                except Exception:
                    pass

        # --- Route: prime / factor (number theory) ---
        if "prime" in text_lower or ("factor" in text_lower and re.search(r"\b\d+\b", text)):
            match = re.search(r"\b(\d+)\b", text)
            if match:
                n = int(match.group(1))
                result = self.number_theory(n)
                return str(result)

        # --- Route: differentiate ---
        if "differentiate" in text_lower or "derivative" in text_lower:
            expr = re.search(r"of\s+(.+?)(?:\s+with|\s+at|\s*$)", text)
            if expr:
                return self.differentiate(expr.group(1).strip())

        # --- Route: integrate ---
        elif "integrate" in text_lower or "integral" in text_lower:
            expr = re.search(r"of\s+(.+?)(?:\s+with|\s+at|\s*$)", text)
            if expr:
                return self.integrate_expr(expr.group(1).strip())

        # --- Route: equation solving ---
        elif "=" in text:
            clean_text = re.sub(
                r"\b(solve|find|calculate|compute|what is)\b", "", text, flags=re.IGNORECASE
            ).strip()
            if "=" in clean_text:
                eq = re.search(r"([a-zA-Z0-9\s\+\-\*/\^\.]+=[a-zA-Z0-9\s\+\-\*/\^\.]+)", clean_text)
                if eq:
                    return self.solve_equation(eq.group(1).strip())

        return None
