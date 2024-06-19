    def update_priorities(self, indices, errors, offset=0.1):
        """Update priorities based on prediction errors."""
        if not isinstance(indices, list) or not isinstance(errors, list):
            raise ValueError("Indices and errors must be passed as lists.")

        if len(indices) != len(errors):
            raise ValueError("The length of indices must match the length of errors.")

        for idx, error in zip(indices, errors):
            if not isinstance(idx, int):
                raise TypeError("Each index must be an integer, got type: {}".format(type(idx)))

            # Ensure error is a number and calculate the new priority
            if not isinstance(error, (int, float)):
                raise TypeError("Each error must be a number, got type: {}".format(type(error)))

            # Calculate new priority and update
            new_priority = (error + offset) ** self.alpha
            if idx >= len(self.priorities):
                raise IndexError("Index out of range. Max allowed index is {}, got {}".format(len(self.priorities)-1, idx))
            self.priorities[idx] = new_priority