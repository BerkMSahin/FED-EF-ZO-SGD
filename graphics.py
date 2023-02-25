import numpy as np
import turtle


class GUI:
    def __init__(self, width, height, simulation):
        self.width = width
        self.height = height
        self.simulation = simulation
        self.agents_an = []
        self.sources_an = []

    # Given a pair of lists of coordinates, creates and animates an agent and a source
    # following the specified trajectories
    def animate(self, agent_locs, source_locs, sources):
        # Screen settings
        drawing_area = turtle.Screen()
        drawing_area.setup(width=self.width, height=self.height)
        drawing_area.bgcolor("black")
        drawing_area.title("Simulation")
        drawing_area.tracer(50, 5) # 50 10

        agent_locs = np.round(agent_locs, 2)
        source_locs = np.round(source_locs, 2)

        for j in range(self.simulation.n):
            # Initialize the agent animation
            animation1 = turtle.Turtle()
            animation1.shape("circle")
            animation1.shapesize(0.25, 0.25, 1)
            animation1.penup()
            animation1.color("green")
            animation1.setx(agent_locs[j, 0, 0])
            animation1.sety(agent_locs[j, 0, 1])
            animation1.speed(10) # 10
            self.agents_an.append(animation1)

            # Initialize the source animation
            animation2 = turtle.Turtle()
            animation2.shape("circle")
            animation2.shapesize(0.25, 0.25, 1)
            animation2.penup()
            animation2.color("red")
            animation2.setx(source_locs[j, 0, 0])
            animation2.sety(source_locs[j, 0, 1])
            animation2.speed(1) #1
            self.sources_an.append(animation2)

        for j in range(1, self.simulation.steps):
            for k in range(self.simulation.n):
                agent_an = self.agents_an[k]
                source_an = self.sources_an[k]

                source_an.goto(np.round(source_locs[k, j], 2))
                agent_an.goto(np.round(agent_locs[k, j], 2))

        turtle.done()
"""
radius = sources[0].radius
        s1, s2, s3 = source_locs[0, 0, :], source_locs[0, 0, :], source_locs[0, 0, :]
        # trajectory 1
        turtle.penup()
        turtle.goto(s1[0] - radius, s1[1])
        turtle.pendown()
        turtle.color("blue")
        turtle.circle(radius)
        # trajectory 2
        turtle.penup()
        turtle.goto(s2[0] - radius, s2[1])
        turtle.pendown()
        turtle.color("cyan")
        turtle.circle(radius)
        # trajectory 3
        turtle.penup()
        turtle.goto(s3[0] - radius, s3[1])
        turtle.pendown()
        turtle.color("yellow")
        turtle.circle(radius)
"""