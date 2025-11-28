import time

created_at = float(time.time())


class SLAAwarePriorityCalculator:
    def calculate_priority(
        self, created_at: float, num_prompts: int, deadline_hours: float = 24.0
    ) -> int:
        """Calculate priority based on task type and hours remaining until deadline."""
        elapsed_hours = (time.time() - created_at) / 3600
        remaining_hours = deadline_hours - elapsed_hours

        # Base priority based on time remaining
        if remaining_hours < 0:
            base_priority = 10  # CRITICAL - deadline already passed!
        elif remaining_hours < 2:
            base_priority = 10  # URGENT - less than 2 hours
        elif remaining_hours < 6:
            base_priority = 8  # HIGH - less than 6 hours
        elif remaining_hours < 12:
            base_priority = 5  # MEDIUM
        else:
            base_priority = 3  # NORMAL - plenty of time

        # Adjust for batch size
        if num_prompts > 500:
            base_priority += 1  # Increase priority for large batches

        return min(base_priority, 10)


if __name__ == "__main__":
    calculator = SLAAwarePriorityCalculator()

    test_job = {
        "job_id": "test_123",
        "created_at": time.time() - 3600,  # 1 hour ago
        "num_prompts": 250,
        "priority": 5,
    }

    priority = calculator.calculate_priority(
        test_job["created_at"], test_job["num_prompts"], deadline_hours=24.0
    )
    print(f"Calculated priority: {priority}")

    # Test with different scenarios
    print("\n--- Testing Different Scenarios ---")

    # Scenario 1: Job just submitted
    now = time.time()
    p1 = calculator.calculate_priority(now, 100, 24.0)
    print(f"Just submitted (100 prompts): priority={p1} (expected: 3)")

    # Scenario 2: Job with 5 hours remaining
    five_hours_ago = now - (19 * 3600)
    p2 = calculator.calculate_priority(five_hours_ago, 100, 24.0)
    print(f"19 hours old / 5h remaining (100 prompts): priority={p2} (expected: 8)")

    # Scenario 3: Job with 1 hour remaining
    twenty_three_hours_ago = now - (23 * 3600)
    p3 = calculator.calculate_priority(twenty_three_hours_ago, 100, 24.0)
    print(f"23 hours old / 1h remaining (100 prompts): priority={p3} (expected: 10)")

    # Scenario 4: Large batch with 10 hours remaining
    fourteen_hours_ago = now - (14 * 3600)
    p4 = calculator.calculate_priority(fourteen_hours_ago, 600, 24.0)
    print(
        f"14 hours old / 10h remaining (600 prompts): priority={p4} (expected: 6 = 5+1)"
    )

    # Scenario 5: Deadline already passed
    twenty_five_hours_ago = now - (25 * 3600)
    p5 = calculator.calculate_priority(twenty_five_hours_ago, 100, 24.0)
    print(f"25 hours old / OVERDUE (100 prompts): priority={p5} (expected: 10)")
