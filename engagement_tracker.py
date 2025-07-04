import os
import pandas as pd
import datetime

class EngagementTracker:
    def __init__(self, output_dir="output", segment_size=20):
        """
        Initialize the EngagementTracker.
        :param output_dir: Directory to store the output Excel file.
        :param segment_size: Number of scores to group into a segment.
        """
        self.output_dir = output_dir
        self.segment_size = segment_size
        self.scores = []
        self.video_name = None

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_video_name(self, video_name):
        """
        Set the name of the video being processed.
        :param video_name: Name of the video file.
        """
        self.video_name = os.path.splitext(os.path.basename(video_name))[0]

    def add_score(self, score):
        """
        Add an engagement score to the tracker.
        :param score: Engagement score (float between 0 and 1).
        """
        self.scores.append(score)

    def calculate_segments(self):
        """
        Calculate average scores for each segment.
        :return: List of segment averages.
        """
        segments = []
        for i in range(0, len(self.scores), self.segment_size):
            segment = self.scores[i:i + self.segment_size]
            segment_avg = sum(segment) / len(segment)
            segments.append(segment_avg)
        return segments

    def calculate_final_score(self):
        """
        Calculate the overall average engagement score.
        :return: Final average score (float).
        """
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def save_to_excel(self):
        """
        Save the engagement scores and segment averages to an Excel file.
        """
        if not self.video_name:
            self.video_name = "unknown_video"

        # Calculate segments and final score
        segments = self.calculate_segments()
        final_score = self.calculate_final_score()

        # Determine rating based on the final score
        if final_score >= 0.8:
            rating = "Excellent"
        elif final_score >= 0.6:
            rating = "Good"
        elif final_score >= 0.4:
            rating = "Average"
        elif final_score >= 0.2:
            rating = "Poor"
        else:
            rating = "Very Poor"

        # Prepare data for Excel
        data = {
            "Score Index": list(range(1, len(self.scores) + 1)),
            "Engagement Score": self.scores
        }
        df_scores = pd.DataFrame(data)

        # Add segment averages
        segment_data = {
            "Segment Index": list(range(1, len(segments) + 1)),
            "Segment Average": segments
        }
        df_segments = pd.DataFrame(segment_data)

        # Add timestamp to the file name to make it unique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{self.video_name}_engagement_{timestamp}.xlsx")

        # Save to Excel
        with pd.ExcelWriter(output_file) as writer:
            df_scores.to_excel(writer, sheet_name="Scores", index=False)
            df_segments.to_excel(writer, sheet_name="Segments", index=False)
            summary_data = {
                "Final Average Score": [final_score],
                "Rating": [rating]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Print the final score and rating as a percentage
        print(f"Final Engagement Score: {final_score * 100:.2f}% ({rating})")
        print(f"Engagement data saved to {output_file}")