"""
BaseModel Animation - Manim Scene
==================================

This script generates an animation showing:
1. LogisticModel box appearing
2. Arrow pointing back to BaseModel
3. Methods appearing from BaseModel (fit, personalize, estimate)
4. BaseModel expanding to show attributes (dimension, features, is_initialized)

Usage:
    poetry run manim -pql basemodel_animation.py BaseModelScene

Output:
    Will be saved to docs/_static/images/animations/
"""

from manim import *

class BaseModelScene(Scene):
    def construct(self):
        # Title
        title = Text("BaseModel: The Foundation", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # 1. LogisticModel appears
        logistic_box = Rectangle(
            height=1.5, 
            width=3, 
            fill_color=BLUE_E, 
            fill_opacity=0.7,
            stroke_color=BLUE
        )
        logistic_text = Text("LogisticModel", font_size=24, color=WHITE)
        logistic_group = VGroup(logistic_box, logistic_text).shift(RIGHT * 3)
        
        self.play(FadeIn(logistic_group, shift=DOWN))
        self.wait(0.5)
        
        # 2. BaseModel appears
        base_box = Rectangle(
            height=1.5, 
            width=3, 
            fill_color=ORANGE, 
            fill_opacity=0.7,
            stroke_color=YELLOW
        )
        base_text = Text("BaseModel", font_size=24, color=WHITE)
        base_group = VGroup(base_box, base_text).shift(LEFT * 3)
        
        self.play(FadeIn(base_group, shift=DOWN))
        self.wait(0.5)
        
        # 3. Arrow pointing from LogisticModel to BaseModel (inheritance)
        arrow = Arrow(
            logistic_group.get_left(), 
            base_group.get_right(), 
            buff=0.1,
            color=GREEN,
            stroke_width=6
        )
        arrow_label = Text("inherits from", font_size=18, color=GREEN).next_to(arrow, UP)
        
        self.play(GrowArrow(arrow), Write(arrow_label))
        self.wait(0.5)
        
        # 4. Methods appear from BaseModel
        methods_title = Text("Core Methods:", font_size=20, color=YELLOW).next_to(base_group, DOWN, buff=0.5)
        
        method_fit = Text("• fit()", font_size=18, color=WHITE).next_to(methods_title, DOWN, buff=0.2)
        method_personalize = Text("• personalize()", font_size=18, color=WHITE).next_to(method_fit, DOWN, buff=0.1)
        method_estimate = Text("• estimate()", font_size=18, color=WHITE).next_to(method_personalize, DOWN, buff=0.1)
        
        methods_group = VGroup(methods_title, method_fit, method_personalize, method_estimate)
        
        self.play(Write(methods_title))
        self.wait(0.3)
        self.play(Write(method_fit))
        self.wait(0.2)
        self.play(Write(method_personalize))
        self.wait(0.2)
        self.play(Write(method_estimate))
        self.wait(0.5)
        
        # 5. BaseModel expands and shows attributes
        # Fade out methods and arrow
        self.play(
            FadeOut(methods_group),
            FadeOut(arrow),
            FadeOut(arrow_label),
            FadeOut(logistic_group)
        )
        
        # Move BaseModel to center and expand
        self.play(base_group.animate.move_to(ORIGIN))
        self.play(base_group.animate.scale(1.5))
        self.wait(0.3)
        
        # Show attributes
        attrs_title = Text("Attributes:", font_size=22, color=YELLOW).next_to(base_group, DOWN, buff=0.7)
        
        attr_dimension = Text("• dimension: int", font_size=18, color=WHITE).next_to(attrs_title, DOWN, buff=0.2)
        attr_features = Text("• features: list[str]", font_size=18, color=WHITE).next_to(attr_dimension, DOWN, buff=0.1)
        attr_initialized = Text("• is_initialized: bool", font_size=18, color=WHITE).next_to(attr_features, DOWN, buff=0.1)
        
        attrs_group = VGroup(attrs_title, attr_dimension, attr_features, attr_initialized)
        
        self.play(Write(attrs_title))
        self.wait(0.3)
        self.play(Write(attr_dimension))
        self.wait(0.2)
        self.play(Write(attr_features))
        self.wait(0.2)
        self.play(Write(attr_initialized))
        self.wait(1)
        
        # Final message
        final_text = Text(
            "BaseModel initializes structure\\nand defines interface",
            font_size=20,
            color=GREEN
        ).to_edge(DOWN, buff=0.5)
        
        self.play(Write(final_text))
        self.wait(2)


# Configuration for output
config.output_file = "../../_static/images/animations/basemodel_animation"
config.media_dir = "../../_static/images/animations"
