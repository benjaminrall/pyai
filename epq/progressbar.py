import time

class ProgressBar:
    def __init__(self, title, maxValue, animationDelay = 0.1, animation = ['|', '/', '-', '\\', '|', '/', '-', '\\'], estimateDelay = 0.1):
        self.title = title
        self.maxValue = maxValue
        self.previousTime = time.time()
        self.estimate = None
        self.estimateDelay = estimateDelay
        self.estimateTime = 0
        self.animation = animation
        self.animationIndex = 0
        self.animationDelay = animationDelay
        self.animationTime = 0
        self.elapsedTime = 0

    # Updates the progress bar for a new specified value
    def update(self, value):
        # Calculates the time that has elapsed since the last update and adds it to necessary time counters
        elapsed = time.time() - self.previousTime
        self.previousTime = time.time()
        self.elapsedTime += elapsed
        self.animationTime += elapsed
        self.estimateTime += elapsed

        # Increments the animation if enough time has passed since the last increment
        if self.animationTime > self.animationDelay:
            self.animationIndex = (self.animationIndex + 1) % len(self.animation)
            self.animationTime = 0

        # Recalculate the estimate if enough time has passed since the last calculation, 
        # otherwise decreases the estimate time by how much time has elapsed
        if self.estimateTime > self.estimateDelay or self.estimate == None:
            self.estimate = (self.elapsedTime / value) * (self.maxValue - value)
            self.estimateTime = 0
        elif self.estimate > 0:
            self.estimate -= elapsed

        # Formats and displays the progress bar
        if value != self.maxValue:
            percentage = (value / self.maxValue) * 100
            progressBars = round(percentage // 5)
            progress = "[" + "=" * (progressBars) + ">" + "." * (20 - progressBars) + "]"
            print("{:^5s}{:<10s} | {:<15s}{:^25s} {}/{} | {}{:<50s}".format(
                self.animation[self.animationIndex],
                self.title,
                f"Progress: {round(percentage, 1)}%",
                progress,
                value,
                self.maxValue,
                f"Time elapsed: {round(self.elapsedTime, 1)}s",
                f" | Estimated time: {round(self.estimate, 1)}s"
            ), end="\r")
        else:
            print("{:^5s}{:<10s} | {:<15s}{:^25s} {}/{} | {}{:<50s}".format(
                "-",
                self.title,
                f"Progress: 100%",
                "[=====================]",
                value,
                self.maxValue,
                f"Time elapsed: {round(self.elapsedTime, 1)}s", ""
            ))