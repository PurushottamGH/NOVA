from sympy import (
    symbols, solve, diff, integrate, 
    simplify, expand, factor, latex,
    Symbol, parse_expr
)
from sympy.parsing.sympy_parser import parse_expr
import re
from typing import Union

class NovaMathEngine:
    """
    Symbolic math solver plugged into Nova.
    NovaMind detects math intent → hands off here → returns exact answer.
    """
    
    def solve_equation(self, equation_str: str) -> str:
        try:
            # Handle symbols beyond just 'x' to be more flexible
            # But stick to symbols('x') for basic solver as per snippet
            x = symbols('x')
            # Handle "x^2 + 2x - 3 = 0" format
            if '=' in equation_str:
                left, right = equation_str.split('=')
                expr = parse_expr(left.strip()) - parse_expr(right.strip())
            else:
                expr = parse_expr(equation_str)
            
            solutions = solve(expr, x)
            return f"Solutions: {solutions}"
        except Exception as e:
            return f"Could not solve: {e}"
    
    def differentiate(self, expr_str: str, var: str = 'x') -> str:
        try:
            x = Symbol(var)
            expr = parse_expr(expr_str)
            result = diff(expr, x)
            return f"d/d{var}({expr_str}) = {result}"
        except Exception as e:
            return f"Could not differentiate: {e}"
    
    def integrate_expr(self, expr_str: str, var: str = 'x') -> str:
        try:
            x = Symbol(var)
            expr = parse_expr(expr_str)
            result = integrate(expr, x)
            return f"∫({expr_str})d{var} = {result} + C"
        except Exception as e:
            return f"Could not integrate: {e}"
    
    def detect_and_solve(self, text: str) -> Union[str, None]:
        """
        Auto-detect if user input contains math and solve it.
        Returns answer string or None if no math detected.
        """
        math_keywords = [
            'solve', 'differentiate', 'integrate', 
            'simplify', 'factor', 'expand', 'derivative',
            'integral', 'equation'
        ]
        
        text_lower = text.lower()
        has_math = any(kw in text_lower for kw in math_keywords)
        
        # Also detect expressions like "x^2 + 3x - 4 = 0"
        has_equation = bool(re.search(r'[a-zA-Z]\^?\d*\s*[+\-*/=]', text))
        
        if not (has_math or has_equation):
            return None
        
        # Route to appropriate solver
        if 'differentiate' in text_lower or 'derivative' in text_lower:
            expr = re.search(r'of\s+(.+?)(?:\s+with|\s+at|\s*$)', text)
            if expr:
                return self.differentiate(expr.group(1).strip())
        
        elif 'integrate' in text_lower or 'integral' in text_lower:
            expr = re.search(r'of\s+(.+?)(?:\s+with|\s+at|\s*$)', text)
            if expr:
                return self.integrate_expr(expr.group(1).strip())
        
        elif '=' in text:
            # Extract equation
            eq = re.search(r'([^:]+=[^:]+)', text)
            if eq:
                return self.solve_equation(eq.group(1).strip())
        
        return None
