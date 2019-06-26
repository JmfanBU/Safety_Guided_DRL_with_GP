import pydart2 as pydart
import numpy as np

# custom pydart world
class DartWorld(pydart.World):
    def __init__(self, *args, **kwargs):
        pydart.World.__init__(self, *args, **kwargs)
        self.arrows = [] # [from, to]

    def on_key_press(self, key):
        pass

    def render_with_ri(self, ri):
        for arrow in self.arrows:
            p0 = arrow[0]
            p1 = arrow[1]
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.025, head_width=0.05, head_len=0.1)

    def reset(self):
        self.arrows = []
        pydart.World.reset(self)