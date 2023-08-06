import time
from collections.abc import Iterable

class ProgressBar:
    """Decorates an iterable object with a progress bar that is printed to the console.
    
    TODO: 
    - Make estimate and animation run on a separate thread 
    - Separate each display aspect and have them combined automatically
    """

    def __init__(self, title: str, iterable: Iterable, minimum_interval: float = 0.01, bars: int = 50,
                 newline_close: bool = True, estimate_interval: float = 0.01, animation_interval: float = 0.1, 
                 animation: list[str] = ['-', '\\', '|', '/', '-', '\\', '|', '/']):
        # Stores iterable to be decorated and finds length of it
        self.iterable = iterable
        self.total = len(iterable)
        self.title = title
        self.bars = bars
        self.bar_percentage = 100 / bars
        self.newline_close = newline_close

        # Time estimate attributes
        self.estimate = float("inf")
        self.estimate_interval = estimate_interval

        # Animation attributes
        self.animation = animation
        self.animation_index = 0
        self.animation_interval = animation_interval
        
        # Current and previous iterator values
        self.n = 0
        self.last_n = 0

        # Minimum intervals and iterations for a display update to occur
        self.minimum_interval = minimum_interval
        self.minimum_iterations = 0

        # Stores local reference to time.time() method and the total time elapsed
        self._time = time.time
        self.time_elapsed = 0
    
    def __iter__(self):
        """Iterable interface for the progress bar."""
        # Stores instance variables as locals for speed optimisation
        iterable = self.iterable
        minimum_interval = self.minimum_interval
        estimate_interval = self.estimate_interval
        animation_interval = self.animation_interval

        # Stores values for tracking the iteration
        n = self.n
        last_n = self.last_n
        total = self.total

        # Stores initial time values
        current_time = self._time()
        last_print_time = current_time
        last_estimate_time = current_time
        last_animation_time = current_time
        start_time = current_time

        # Loops through the iterable
        for obj in iterable:
            # Returns next object from the iterable
            yield obj

            # Increments n
            n += 1

            # Checks if minimum iterations condition for updating is met
            if n - last_n >= self.minimum_iterations:
                current_time = self._time()
                
                # Checks if the progress bar should be updated
                print_difference = current_time - last_print_time
                if print_difference >= minimum_interval:
                    self.update(n - last_n)
                    last_n = self.last_n
                    last_print_time += print_difference
                    self.time_elapsed = current_time - start_time

                # Checks if the estimate should be recalculated
                estimate_difference = current_time - last_estimate_time
                if estimate_difference >= estimate_interval:
                    self.estimate = ((current_time - start_time) / n) * (total - n)
                    last_estimate_time += estimate_difference
        
                # Checks if the animation should progress
                animation_difference = current_time - last_animation_time
                if animation_difference >= animation_interval:
                    self.animation_index = (self.animation_index + 1) % len(self.animation)
                    last_animation_time += animation_difference

        # Closes the progress bar
        self.close()

    def __len__(self):
        return len(self.iterable)

    def update(self, change=1):
        """Manually updates the progress bar with a specified change."""
        # Does not allow negative or 0 change
        if change <= 0:
            return
        
        # Updates the iterator position
        self.n += change
        dn = self.n - self.last_n 

        # Checks if the progress bar being displayed needs to be updated
        if dn >= self.minimum_iterations:
            # Updates last n and displays the progress bar
            self.last_n = self.n
            self.display()
      
            # Calculates the new minimum iterations before updating
            self.minimum_iterations = max(self.minimum_iterations, dn)

    def display(self):
        """Prints the current progress bar and information."""
        # Calculates current percentage
        percentage = self.n / self.total

        # Calculates amount of filled bars and uses it to construct progress bars
        bars_filled = int(percentage * self.bars)
        progress = '[' + '=' * bars_filled + '>' + '.' * (self.bars - bars_filled - 1) + ']'

        # Prints full formatted string
        print("{:^5s}{:s} - {:>6.1%} {:^s} {:{}d}/{:d} - Time elapsed: {:.1f}s - Estimate time: {:.1f}s{:10s}".format(
            self.animation[self.animation_index], self.title, percentage, 
            progress, self.n, len(str(self.total)), self.total, self.time_elapsed, self.estimate, ''
        ), end='\r')

    def close(self):
        """Disables the progress bar and prints final display."""
        # Prints over last display with complete closed formatted progress bar string
        print("{:^5s}{:s} - {:>6.1%} {:^s} {}/{} - Time elapsed: {:.1f}s".format(
            self.animation[0], self.title, 1, '[' + '=' * self.bars + ']', 
            self.total, self.total, self.time_elapsed
        ), end = '\n' if self.newline_close else '')
