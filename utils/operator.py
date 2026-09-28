"""
Threshold-based comparison operators for floating-point values.

Used in musical timing where tiny floating-point differences should be
considered equal for practical purposes (e.g., MIDI timing precision).
"""


class Operator:
    """
    Comparison operators with a threshold for treating nearly-equal values as equal.
    
    This is useful for comparing musical timing values where floating-point
    precision might cause tiny differences that aren't musically significant.
    
    Example:
        >>> op = Operator(threshold=0.01)
        >>> op.equal(1.0, 1.005)  # True - difference is 0.005 <= 0.01
        >>> op.equal(1.0, 1.02)   # False - difference is 0.02 > 0.01
        >>> op.greater(1.02, 1.0) # True - 1.02 is significantly greater
        >>> op.greater(1.005, 1.0) # False - within threshold, considered equal
    """
    
    def __init__(self, threshold: float = 1):
        """
        Initialize the operator with a comparison threshold.
        
        Args:
            threshold: Maximum difference to consider two values equal.
        """
        self.threshold = abs(threshold)  # Ensure positive
    
    def equal(self, a: float, b: float, threshold: float = None) -> bool:
        """
        Check if two values are equal within the threshold.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if |a - b| <= threshold
        """
        return abs(a - b) <= self.threshold if threshold is None else abs(a - b) <= abs(threshold)
    
    def not_equal(self, a: float, b: float) -> bool:
        """
        Check if two values are not equal (difference exceeds threshold).
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if |a - b| > threshold
        """
        return abs(a - b) > self.threshold
    
    def greater(self, a: float, b: float, threshold: float = None) -> bool:
        """
        Check if a is significantly greater than b.
        
        Values within threshold are considered equal, so a must be
        greater than b by more than the threshold.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if a - b > threshold
        """
        return a - b > self.threshold if threshold is None else a - b > abs(threshold)
    
    def less(self, a: float, b: float, threshold: float = None) -> bool:
        """
        Check if a is significantly less than b.
        
        Values within threshold are considered equal, so b must be
        greater than a by more than the threshold.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if b - a > threshold
        """
        return b - a > self.threshold if threshold is None else b - a > abs(threshold)
    
    def greater_or_equal(self, a: float, b: float, threshold: float = None) -> bool:
        """
        Check if a is greater than or equal to b (within threshold).
        
        Returns True if a > b OR a ≈ b.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if a >= b - threshold
        """
        return a >= b - self.threshold if threshold is None else a >= b - abs(threshold)
    
    def less_or_equal(self, a: float, b: float, threshold: float = None) -> bool:
        """
        Check if a is less than or equal to b (within threshold).
        
        Returns True if a < b OR a ≈ b.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            True if a <= b + threshold
        """
        return a <= b + self.threshold if threshold is None else a <= b + abs(threshold)
    
    # Alias methods for convenience
    def eq(self, a: float, b: float) -> bool:
        """Alias for equal()."""
        return self.equal(a, b)
    
    def ne(self, a: float, b: float) -> bool:
        """Alias for not_equal()."""
        return self.not_equal(a, b)
    
    def gt(self, a: float, b: float) -> bool:
        """Alias for greater()."""
        return self.greater(a, b)
    
    def lt(self, a: float, b: float) -> bool:
        """Alias for less()."""
        return self.less(a, b)
    
    def ge(self, a: float, b: float) -> bool:
        """Alias for greater_or_equal()."""
        return self.greater_or_equal(a, b)
    
    def le(self, a: float, b: float) -> bool:
        """Alias for less_or_equal()."""
        return self.less_or_equal(a, b)