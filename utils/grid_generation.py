import math

def create_regions(size, num_regions):
    # Check if num_regions is a perfect square
    if int(math.sqrt(num_regions)) ** 2 != num_regions:
        raise ValueError("Number of regions must be a perfect square.")
    
    rows = cols = int(math.sqrt(num_regions))
    step_x = size // cols
    step_y = size // rows
    
    REGION_BOUND = {}
    id = 1
    for row in range(rows):
        for col in range(cols):
            # Calculate top-left and bottom-right corners of each region
            start_x = col * step_x
            start_y = row * step_y
            end_x = start_x + step_x - 1
            end_y = start_y + step_y - 1
            
            # If on the last column or last row, adjust to fit exactly to size
            if col == cols - 1:
                end_x = size - 1
            if row == rows - 1:
                end_y = size - 1
            
            REGION_BOUND[id] = ((start_x, start_y), (end_x, end_y))
            id += 1
    return REGION_BOUND
