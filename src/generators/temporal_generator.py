"""
Temporal Pattern Generator for SynFinance
Week 2, Day 1-2: Time-Based Patterns

Generates realistic temporal patterns for Indian financial transactions:
- Hour-of-day distributions (breakfast, lunch, evening peaks)
- Day-of-week patterns (weekday vs weekend behavior)
- Monthly patterns (salary day spikes on 1st, 30th)
- Festival patterns (Diwali, Holi, Eid, Christmas spending boosts)

Author: SynFinance Team
Date: October 18, 2025
Status: Week 2 Implementation
"""

import random
from datetime import datetime
from typing import List, Dict, Tuple
from src.customer_profile import CustomerProfile, CustomerSegment, Occupation


class TemporalPatternGenerator:
    """
    Generates temporal patterns for transactions based on:
    - Customer occupation (work schedules)
    - Customer segment (lifestyle patterns)
    - Day of week (weekday vs weekend)
    - Day of month (salary days)
    - Festivals and holidays
    """
    
    # Hour-of-day probability distributions by occupation
    OCCUPATION_HOUR_DISTRIBUTIONS = {
        Occupation.SALARIED_EMPLOYEE: {
            # Peak hours: Before work (7-9am), lunch (12-2pm), after work (6-10pm)
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5am: Very rare
                0.05, 0.08, 0.06, 0.03, 0.02, 0.02,  # 6-11am: Morning peak before work
                0.06, 0.08, 0.04, 0.02, 0.02, 0.03,  # 12-5pm: Lunch peak, low afternoon
                0.08, 0.10, 0.12, 0.10, 0.06, 0.02   # 6-11pm: Evening peak after work
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5am: Very rare
                0.02, 0.03, 0.05, 0.06, 0.08, 0.09,  # 6-11am: Gradual rise
                0.08, 0.08, 0.07, 0.06, 0.05, 0.06,  # 12-5pm: Afternoon shopping
                0.07, 0.08, 0.07, 0.05, 0.03, 0.02   # 6-11pm: Evening activity
            ]
        },
        Occupation.BUSINESS_OWNER: {
            # More flexible, spread throughout day
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: Very rare
                0.04, 0.06, 0.07, 0.06, 0.05, 0.05,  # 6-11am: Morning activity
                0.06, 0.07, 0.06, 0.05, 0.05, 0.06,  # 12-5pm: Afternoon steady
                0.07, 0.08, 0.08, 0.07, 0.04, 0.02   # 6-11pm: Evening activity
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: Very rare
                0.03, 0.04, 0.05, 0.06, 0.07, 0.08,  # 6-11am: Morning rise
                0.08, 0.08, 0.07, 0.06, 0.05, 0.06,  # 12-5pm: Afternoon
                0.07, 0.08, 0.07, 0.05, 0.03, 0.02   # 6-11pm: Evening
            ]
        },
        Occupation.FREELANCER: {
            # Very flexible, late mornings and evenings
            "weekday": [
                0.02, 0.02, 0.02, 0.01, 0.01, 0.02,  # 0-5am: Night owls
                0.03, 0.04, 0.05, 0.06, 0.07, 0.07,  # 6-11am: Late morning
                0.07, 0.07, 0.06, 0.05, 0.05, 0.06,  # 12-5pm: Afternoon
                0.07, 0.08, 0.08, 0.07, 0.05, 0.03   # 6-11pm: Evening peak
            ],
            "weekend": [
                0.02, 0.02, 0.02, 0.01, 0.01, 0.02,  # 0-5am: Night owls
                0.03, 0.04, 0.05, 0.06, 0.07, 0.08,  # 6-11am: Late morning
                0.08, 0.08, 0.07, 0.06, 0.05, 0.06,  # 12-5pm: Afternoon
                0.07, 0.08, 0.07, 0.05, 0.03, 0.02   # 6-11pm: Evening
            ]
        },
        Occupation.STUDENT: {
            # Classes during day, free in evening
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5am: Very rare
                0.02, 0.03, 0.02, 0.02, 0.03, 0.04,  # 6-11am: Morning classes
                0.05, 0.05, 0.04, 0.04, 0.05, 0.07,  # 12-5pm: Lunch & afternoon
                0.09, 0.11, 0.12, 0.10, 0.08, 0.04   # 6-11pm: Evening free time (PEAK)
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5am: Very rare
                0.02, 0.03, 0.04, 0.06, 0.08, 0.09,  # 6-11am: Late wake up
                0.08, 0.08, 0.07, 0.06, 0.06, 0.07,  # 12-5pm: Afternoon
                0.08, 0.09, 0.08, 0.06, 0.04, 0.02   # 6-11pm: Evening
            ]
        },
        Occupation.HOMEMAKER: {
            # Morning and afternoon shopping, cooking times
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5am: Early risers
                0.06, 0.08, 0.09, 0.08, 0.07, 0.06,  # 6-11am: Morning shopping (PEAK)
                0.06, 0.05, 0.05, 0.06, 0.07, 0.06,  # 12-5pm: Afternoon errands
                0.06, 0.06, 0.05, 0.03, 0.02, 0.01   # 6-11pm: Evening cooking
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5am: Early risers
                0.05, 0.07, 0.08, 0.08, 0.08, 0.07,  # 6-11am: Morning
                0.07, 0.07, 0.06, 0.06, 0.06, 0.06,  # 12-5pm: Afternoon
                0.06, 0.06, 0.05, 0.03, 0.02, 0.01   # 6-11pm: Evening
            ]
        },
        Occupation.RETIRED: {
            # Early morning routine, avoid evening
            "weekday": [
                0.01, 0.01, 0.01, 0.02, 0.03, 0.05,  # 0-5am: Early risers
                0.08, 0.09, 0.09, 0.08, 0.07, 0.06,  # 6-11am: Morning routine (PEAK)
                0.06, 0.06, 0.05, 0.05, 0.04, 0.04,  # 12-5pm: Afternoon
                0.04, 0.03, 0.02, 0.02, 0.01, 0.01   # 6-11pm: Early evening, then home
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.02, 0.03, 0.05,  # 0-5am: Early risers
                0.08, 0.09, 0.09, 0.08, 0.07, 0.06,  # 6-11am: Morning
                0.06, 0.06, 0.05, 0.05, 0.04, 0.04,  # 12-5pm: Afternoon
                0.04, 0.03, 0.02, 0.02, 0.01, 0.01   # 6-11pm: Evening
            ]
        },
        Occupation.PROFESSIONAL: {
            # Similar to salaried but slightly more flexible
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: Very rare
                0.04, 0.07, 0.06, 0.04, 0.03, 0.03,  # 6-11am: Morning
                0.05, 0.07, 0.05, 0.04, 0.04, 0.05,  # 12-5pm: Lunch & afternoon
                0.08, 0.10, 0.11, 0.09, 0.06, 0.03   # 6-11pm: Evening (PEAK)
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: Very rare
                0.03, 0.04, 0.06, 0.07, 0.08, 0.09,  # 6-11am: Late morning
                0.08, 0.08, 0.07, 0.06, 0.05, 0.06,  # 12-5pm: Afternoon
                0.07, 0.08, 0.07, 0.05, 0.03, 0.02   # 6-11pm: Evening
            ]
        },
        Occupation.GOVERNMENT_EMPLOYEE: {
            # Very regular hours, 9-5 mentality
            "weekday": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5am: Very rare
                0.04, 0.07, 0.05, 0.02, 0.02, 0.02,  # 6-11am: Morning before work
                0.05, 0.07, 0.03, 0.02, 0.02, 0.03,  # 12-5pm: Lunch peak
                0.09, 0.11, 0.12, 0.11, 0.07, 0.03   # 6-11pm: After work (PEAK)
            ],
            "weekend": [
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: Very rare
                0.03, 0.05, 0.06, 0.07, 0.08, 0.09,  # 6-11am: Morning
                0.08, 0.08, 0.07, 0.06, 0.05, 0.06,  # 12-5pm: Afternoon
                0.07, 0.08, 0.07, 0.05, 0.03, 0.02   # 6-11pm: Evening
            ]
        }
    }
    
    # Segment-based day-of-week multipliers
    SEGMENT_WEEKDAY_MULTIPLIERS = {
        CustomerSegment.YOUNG_PROFESSIONAL: {
            "weekday": 0.9,   # Busier on weekdays, save for weekend fun
            "weekend": 1.3    # More spending on weekends (social, dining, entertainment)
        },
        CustomerSegment.FAMILY_ORIENTED: {
            "weekday": 1.0,   # Steady spending (groceries, utilities)
            "weekend": 1.2    # Family outings, shopping
        },
        CustomerSegment.BUDGET_CONSCIOUS: {
            "weekday": 1.0,   # Steady essential spending
            "weekend": 0.95   # Slightly lower on weekends (saving mode)
        },
        CustomerSegment.TECH_SAVVY_MILLENNIAL: {
            "weekday": 0.85,  # Lower on weekdays (working)
            "weekend": 1.4    # High weekend spending (experiences, online shopping)
        },
        CustomerSegment.AFFLUENT_SHOPPER: {
            "weekday": 1.1,   # Can shop anytime
            "weekend": 1.3    # Premium weekend experiences
        },
        CustomerSegment.SENIOR_CONSERVATIVE: {
            "weekday": 1.05,  # Prefer weekdays (less crowded)
            "weekend": 0.9    # Lower on weekends (avoid crowds)
        },
        CustomerSegment.STUDENT: {
            "weekday": 0.8,   # Limited spending during classes
            "weekend": 1.5    # Much higher on weekends (free time)
        }
    }
    
    # Salary day multipliers (1st and 30th of month)
    SALARY_DAY_MULTIPLIERS = {
        CustomerSegment.YOUNG_PROFESSIONAL: 1.8,      # High spending after salary
        CustomerSegment.FAMILY_ORIENTED: 2.0,         # Bill payments, bulk shopping
        CustomerSegment.BUDGET_CONSCIOUS: 1.5,        # Planned purchases
        CustomerSegment.TECH_SAVVY_MILLENNIAL: 1.7,   # Online shopping spree
        CustomerSegment.AFFLUENT_SHOPPER: 1.3,        # Less impact (always have money)
        CustomerSegment.SENIOR_CONSERVATIVE: 1.4,     # Pension day spending
        CustomerSegment.STUDENT: 1.2                  # Allowance/stipend day
    }
    
    # Pre-salary days (28th, 29th) - reduced spending
    PRE_SALARY_MULTIPLIER = 0.7  # 30% reduction before salary
    
    def __init__(self, seed: int = None):
        """
        Initialize temporal pattern generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.random = random.Random(seed)
    
    def get_hour_distribution(
        self,
        customer: CustomerProfile,
        date: datetime
    ) -> List[float]:
        """
        Get hour-of-day probability distribution for customer
        
        Returns list of 24 probabilities (one per hour, 0-23)
        Based on occupation and whether it's weekday/weekend
        
        Args:
            customer: Customer profile
            date: Transaction date
            
        Returns:
            List of 24 floats summing to ~1.0 (probabilities for each hour)
        """
        is_weekend = date.weekday() >= 5  # Saturday=5, Sunday=6
        day_type = "weekend" if is_weekend else "weekday"
        
        # Get base distribution for occupation
        if customer.occupation in self.OCCUPATION_HOUR_DISTRIBUTIONS:
            distribution = self.OCCUPATION_HOUR_DISTRIBUTIONS[customer.occupation][day_type]
        else:
            # Default distribution (generic)
            distribution = self.OCCUPATION_HOUR_DISTRIBUTIONS[Occupation.SALARIED_EMPLOYEE][day_type]
        
        return distribution
    
    def select_transaction_hour(
        self,
        customer: CustomerProfile,
        date: datetime
    ) -> int:
        """
        Select transaction hour based on customer's temporal patterns
        
        Args:
            customer: Customer profile
            date: Transaction date
            
        Returns:
            Hour (0-23)
        """
        distribution = self.get_hour_distribution(customer, date)
        hour = self.random.choices(range(24), weights=distribution)[0]
        return hour
    
    def get_day_of_week_multiplier(
        self,
        customer: CustomerProfile,
        date: datetime
    ) -> float:
        """
        Get spending multiplier based on day of week
        
        Args:
            customer: Customer profile
            date: Transaction date
            
        Returns:
            Multiplier (0.8-1.5)
        """
        is_weekend = date.weekday() >= 5
        day_type = "weekend" if is_weekend else "weekday"
        
        # Check by enum value to avoid import path mismatches
        for segment_key, multipliers in self.SEGMENT_WEEKDAY_MULTIPLIERS.items():
            if segment_key.value == customer.segment.value:
                return multipliers[day_type]
        
        return 1.0  # No change
    
    def is_salary_day(self, date: datetime) -> bool:
        """
        Check if date is a salary day (1st or 30th of month)
        
        Also considers last day of month (28th Feb, 31st in 31-day months)
        
        Args:
            date: Transaction date
            
        Returns:
            True if salary day
        """
        day = date.day
        
        # 1st of month (common salary day)
        if day == 1:
            return True
        
        # Last 2 days of month (30th, 31st, or 28th/29th Feb)
        # Check if it's within 2 days of month end
        if date.month == 12:
            next_month = date.replace(year=date.year + 1, month=1, day=1)
        else:
            next_month = date.replace(month=date.month + 1, day=1)
        
        from datetime import timedelta
        days_until_next_month = (next_month - date).days
        
        return days_until_next_month <= 2  # Last 2 days of month
    
    def is_pre_salary_day(self, date: datetime) -> bool:
        """
        Check if date is 2-3 days before salary (tight budget period)
        
        Args:
            date: Transaction date
            
        Returns:
            True if pre-salary day
        """
        day = date.day
        
        # Days 27-29 are typically pre-salary days
        return 27 <= day <= 29 and not self.is_salary_day(date)
    
    def get_salary_day_multiplier(
        self,
        customer: CustomerProfile,
        date: datetime
    ) -> float:
        """
        Get spending multiplier for salary days
        
        Args:
            customer: Customer profile
            date: Transaction date
            
        Returns:
            Multiplier (0.7-2.0)
        """
        if self.is_salary_day(date):
            # Salary day - increased spending
            if customer.segment in self.SALARY_DAY_MULTIPLIERS:
                return self.SALARY_DAY_MULTIPLIERS[customer.segment]
            else:
                return 1.5  # Default 50% increase
        
        elif self.is_pre_salary_day(date):
            # Pre-salary days - reduced spending (tight budget)
            return self.PRE_SALARY_MULTIPLIER
        
        else:
            # Normal day
            return 1.0
    
    def get_festival_multiplier(
        self,
        date: datetime,
        customer: CustomerProfile,
        festivals: Dict[str, Dict]
    ) -> Tuple[float, str]:
        """
        Get spending multiplier for festivals
        
        Checks if date falls within festival period and returns appropriate multiplier
        
        Args:
            date: Transaction date
            customer: Customer profile
            festivals: Dictionary of festivals with month and multiplier
            
        Returns:
            Tuple of (multiplier, festival_name) or (1.0, "")
        """
        for festival_name, festival_info in festivals.items():
            festival_months = festival_info.get("month")
            
            # Handle both single month and list of months
            if isinstance(festival_months, list):
                months = festival_months
            else:
                months = [festival_months]
            
            if date.month in months:
                # Festival month - apply spending boost
                base_multiplier = festival_info.get("spending_multiplier", 1.5)
                
                # Adjust multiplier based on customer segment
                if customer.segment == CustomerSegment.AFFLUENT_SHOPPER:
                    # Affluent shoppers spend even more during festivals
                    multiplier = base_multiplier * 1.2
                elif customer.segment == CustomerSegment.BUDGET_CONSCIOUS:
                    # Budget conscious spend less during festivals
                    multiplier = base_multiplier * 0.8
                elif customer.segment == CustomerSegment.FAMILY_ORIENTED:
                    # Family oriented spend more (gifts, family gatherings)
                    multiplier = base_multiplier * 1.1
                else:
                    multiplier = base_multiplier
                
                return (multiplier, festival_name)
        
        return (1.0, "")  # No festival
    
    def get_combined_temporal_multiplier(
        self,
        customer: CustomerProfile,
        date: datetime,
        festivals: Dict[str, Dict]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Get combined temporal multiplier from all sources
        
        Combines:
        - Day of week multiplier
        - Salary day multiplier
        - Festival multiplier
        
        Args:
            customer: Customer profile
            date: Transaction date
            festivals: Dictionary of festivals
            
        Returns:
            Tuple of (combined_multiplier, breakdown_dict)
        """
        dow_mult = self.get_day_of_week_multiplier(customer, date)
        salary_mult = self.get_salary_day_multiplier(customer, date)
        festival_mult, festival_name = self.get_festival_multiplier(date, customer, festivals)
        
        # Combine multipliers (multiplicative)
        combined = dow_mult * salary_mult * festival_mult
        
        # Cap at reasonable range (0.5x to 3.0x)
        combined = max(0.5, min(3.0, combined))
        
        breakdown = {
            "day_of_week": dow_mult,
            "salary_day": salary_mult,
            "festival": festival_mult,
            "festival_name": festival_name,
            "combined": combined
        }
        
        return (combined, breakdown)


# Module exports
__all__ = ['TemporalPatternGenerator']
