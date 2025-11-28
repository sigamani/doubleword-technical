import time

def calculate_priority(created_at: float, num_prompts: int, deadline_hours = 24.0) -> int:
    """Calculate priority based on task type and hours remaining until deadline."""
    elapsed_hours = (time.time() - created_at) / 3600# Assume a default deadline of 24 hours
    remaining_hours = deadline_hours - elapsed_hours
    

    # Urgency based on hours remaining
    if remaining_hours < 2:
        base_priority = 10
    elif remaining_hours < 6:
        base_priority = 8
    elif remaining_hours < 12:
        base_priority = 5
    else:
        base_priority = 3
    if num_prompts > 500:
        base_priority += 1  # Increase priority for large batches
    return min(base_priority, 10)

    