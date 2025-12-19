---
title: 'condynsate: A Python-Based Controls and Dynamics Simulation Package'
tags:
  - Python
  - control
  - dynamics
authors: 
  - name: Schaer G. 
    orcid: 0000-0002-0915-8627
    affiliation: 1
  - name: Chang W.
    affiliation: 1
  - name: Eggl S.
    orcid: 0000-0002-1398-6302
    affiliation: 1
  - name: Bretl T.
    orcid: 0000-0001-7883-7300
    affiliation: 1
  - name: Hilgenfeldt S.
    orcid: 0000-0002-6799-2118
    affiliation: 2
affiliations:
 - name: The Grainger College of Engineering, Department of Aerospace Engineering, University of Illinois Urbana-Champaign
   index: 1
 - name: The Grainger College of Engineering, Department of Mechanical Science & Engineering, University of Illinois Urbana-Champaign
   index: 2
bibliography: paper.bib
---

# Summary

We present a Python-based, open-source software package called ``condynsate`` (**con**trol and **dyn**amics simulator) designed to ease the creation of computation-based dynamics and control demonstrations, assignments, and projects. Guided by a simulation and role-play pedagogy, projects built with ``condynsate`` mimics the experience of simulation-based games to enhance learning and retention. 

Users can import rigid and articulated bodies into a simulation environment, visualize the simulation environment in 3D, tweak the appearance of the visualization, create and display live-updating plots, read the state, apply arbitrary forces, and tweak the physics and appearance of all joints and links in each body, run full physics simulations, interact with the environment through keypresses, and record the visual outputs of ``condynsate``, all in real time.

All materials, including the package and example usage, have been made publicly available at [https://github.com/condynsate/condynsate](https://github.com/condynsate/condynsate) and are licensed under the GPL-3.0-only and MIT licenses. To install, type

```bash
python3 -m pip install condynsate
```

in a Python-enabled terminal.

# Statement of Need

Dynamics, mechanical systems, and controls are fundamental topics in the fields of mechanical, aerospace, and robotics engineering [@Greenwood:1988; @Franklin:1986; @Angeles:2014]; however, conventional dynamics and control educational approaches rely heavily upon either classroom-constrained lecture or laboratory demonstrations, both of which can be limited by format, materials, or cost. Informed by the viability of virtual-supplemented education in robotics [@Jaakkola:2008; @Jaakkola:2011; @Berland:2015], and in response to these limitations, we identified a primary need for a classroom-deployable educational tool that promotes student engagement with mechanical systems through simulation, visualization, and interactivity. 

For example, a well-known system with non-intuitive dynamics is the multi-axis gyroscope. Whereas the equations of motion of this system can be derived with ease, the intuition of the dynamic effects of varying parameters and torques is convoluted by the complexity of the governing equations [@GreenwoodChapter:1988]. Similarly challenging is the problem of controlling a quadrotor, a classic aerospace example. Designing the controller itself is a relatively simple task; however, understanding and visualizing how changes to the controller's design induce specific variations in trajectories is much more involved and may prove impenetrable for some students when constrained to classroom environments. By providing a simulation tool that both visualizes systems and facilitates student-system interactivity via real-time parameter variation coupled with keyboard interactivity, students are given the opportunity to observe the consequences of system manipulation via both numerical and visual feedback. This real-time interaction and feedback mimics the experience of simulation-based games, familiar and fun, rather than conventional laboratory demos. This technique, when coupled with structured prompting and reflection, is beneficial to learning and retention [@Dankbaar:2016; @Veermans:2019; @Chernikova:2020].

In addition to the primary need of providing supplemental exposure to complex mechanical systems, we identified a secondary need of continuing computation-based education, specifically in the scope of post-secondary education. Computational thinking (CT) and computational literacy are professional skill sets highly valued in academia, education, and industry [@Weintrop:2016; @Braun:2022]. Indeed, a set of surveys we conducted at the University of Illinois Urbana-Champaign between 2023 and 2025 indicated that approximately 91% of incoming junior-level aerospace engineering students agree that highly developed computational skills are important for their future careers. Therefore, our focus was on the later stage of undergraduate education, where less "recipe-style" and more creative implementations of tools and solutions are presented to students. Built upon foundational programming experience from earlier-stage classes—the same survey showed that approximately 87% of the same students have at least some experience in the Python programming language—we aimed to develop a tool that encourages students to take initiative-driven and imaginative algorithmic approaches while exploring real-world problems, all in a Python-based computational environment.

Finally, given the broad scope of dynamics, mechanical systems, and controls, and coupled with the above philosophy of unique projects promoting original thinking, it would be insufficient to provide only a learning module to address the primary and secondary needs. Consequently, we identified a tertiary need for a generalized approach for educators to make sophisticated lecture demonstrations, assignments, and projects related to fundamental engineering concepts with ease.

Combining the three identified needs of 1) in-classroom exposure to mechanical systems, 2) continuing computation-based education, and 3) a method of development of novel projects, we designed a Python-based dynamic system simulation and visualization tool called ``condynsate`` with the explicit philosophy of not limiting project complexity while simultaneously promoting ease of use.

# The condynsate Package

With a physics engine provided by ``PyBullet`` [@PyBullet] and 3D visualization provided by ``MeshCat`` [@MeshCat], ``condynsate`` implements real-time simulation of .stl and .obj defined rigid bodies and .urdf defined articulated bodies. It allows users to interact with simulation results through a browser-based 3D viewer to visualize simulations, a built-in animator to plot arbitrary simulation states, and a keyboard module that allows detection of key press events. These features equip ``condynsate`` with a broad scope of applicability by guaranteeing that any dynamic system that can be described by a .urdf file, a file format created by Open Robotics for the Robot Operating System software [@ROS], is supported.

Documentation, tutorials, and examples were generated with the intent of educating instructors on the usage of ``condynsate`` to develop teaching demonstrations. The five included tutorials walk through the development of ``condynsate``-based projects, including the simulation environment, design and implementation of mechanical systems, application of internal and external forces and torques to the mechanical systems, real-time visualization and animation, keyboard interactivity, and methods of data collection.

 In addition, at least one example of usage is included for each of the five major modules: the ``Keyboard`` module, which provides keyboard interactivity; the ``Visualizer`` module, which provides 3D visualization; the ``Animator`` module, which provides GUI-based plotting; the ``Simulator`` module, which provides the physics simulation environment; and the ``Project`` module, which provides automatic interfacing with all other modules for easy project development. The ``Project`` module examples best demonstrate how educators may use ``condynsate`` to construct assignments and demonstrations. As such, the provided examples are common mechanical systems used in dynamics and control education: an inverted pendulum autonomously balanced on a 4-wheeled cart, a three-axis gyroscope with keyboard interactivity, and a double pendulum with real-time phase space plotting.

To date, projects built with ``condynsate`` have been successfully deployed in an undergraduate aerospace controls course during the Spring 2024 and Fall 2024 semesters at the University of Illinois Urbana-Champaign.  Post-semester student survey results showed that 69% of students felt better prepared to solve control problems using computational tools, and 86% stated they felt their Python skills improved.  We conclude that the deployment and continued development of computation-based curricula provide an enhanced approach for dynamics and control education. Computationally literate students are better equipped to exploit the full versatility of a computer to tackle complex problems, and we propose that ``condynsate`` can set students on the path of utilizing computational resources as a matter-of-course tool both at university and in their careers.

# Acknowledgements

The development of ``condynsate`` was funded by the [Computational Tools for Dynamics and Control grant](https://ae3.grainger.illinois.edu/programs/siip-grants/64459) through the University of Illinois Urbana-Champaign Grainger College of Engineering Academy for Excellence in Engineering Education (AE3) Strategic Instructional Innovations Program (SIIP).

# References

