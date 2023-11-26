# CS224W
### CS224W: Machine Learning with Graphs by [Jure Leskovec](https://profiles.stanford.edu/jure-leskovec)
- [Course outline](http://snap.stanford.edu/class/cs224w-2021/)

## What is this course about?
Complex data can be represented as a graph of relationships between objects. Such networks are a fundamental tool for modeling social, technological, and biological systems. This course focuses on the computational, algorithmic, and modeling challenges specific to the analysis of massive graphs. By means of studying the underlying graph structure and its features, students are introduced to machine learning techniques and data mining tools apt to reveal insights on a variety of networks.
Topics include representation learning and Graph Neural Networks; algorithms for the World Wide Web; reasoning over Knowledge Graphs; influence maximization; disease outbreak detection, and social network analysis.

- Mr. Jingbo Yang created a synopsis of this course and made it available to the public. I can't read it now, but I hope it will be beneficial later. Check this out:
  - [CS224W Notes from Fall 2019](https://jingboyang.github.io/cs224w_notes.html)

- Because of their reputation, these two persons are incredibly generous individuals who are uncommon in my school's community. What they've done is quite valuable and has encouraged my own *GNN* studying.
  - [SlidesToCode Repo](https://github.com/mnslarcher/cs224w-slides-to-code/blob/main/notebooks/02-traditional-methods-for-ml-on-graphs.ipynb)
  - [GCNfromScratch Repo](https://github.com/jason2133/CS224W)

⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⡿⠁⣿⣿⣿⠋⣰⣿⡿⠁⣾⣿⣿⡿⠃⢸⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡻⣿⣿⣿⣿⣿⡿⢃⣿⣿⠃⢰⡿⢻⠁⢀⡿⠋⠀⡰⣿⣿⠟⠁⠀⣾⣿⠿⣟⣿⡿⠋⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⣿⣿⣿⠟⡟⠁⢸⠏⡈⠀⠸⠁⠇⠀⠸⠀⠀⠐⣰⠟⠁⠀⠀⡸⠋⣰⣾⡿⠋⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⡇⠀⣿⠏⢡⠎⠀⢀⡆⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠘⠋⠁⠀⠀⣠⣿⢟⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⢻⣻⠇⢇⢸⠃⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠊⣉⢴⣽⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⢿⠀⠀⠀⠀⠀⠀⢠⠂⠀⠀⠀⠀⠀⠀⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠔⠛⠉⠉⢉⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠘⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⠻⣟⢹⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠀⠀⢠⢃⠀⠀⠀⠀⠀⠀⡀⠀⠀⣀⡠⠄⠀⠀⠀⠀⠀⠐⠊⠉⠙⠓⠛⣉⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣷⣌⠙⠣⠀⡀⠀⠀⠀⢸⠀⢸⠀⠀⠀⠀⢠⠊⢀⠋⣸⢠⠃⠀⢀⠔⠋⡠⣶⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⠓⢄⡀⢇⠀⠀⠀⠸⡀⠸⡀⠀⠀⠰⠁⡠⠃⣠⡷⠁⢀⢔⠁⢀⡔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⠚⠉⠉⠛⠛⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⡿⢿⣈⣦⠀⠀⠘⣄⠀⠀⠀⢳⡀⠷⣀⣀⠿⠴⢥⣾⡯⠥⠤⠧⠤⠤⠮⠤⡤⢤⠤⡄⣀⣀⣴⡶⠂⠀⠀⠀⠀⣶⣶⣺⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣉⡁⠀⠀⠫⡦⣠⣀⡤⠟⠊⠹⠒⠊⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠓⠂⠀⠉⠣⣖⣦⣄⡀⠈⣡⣤⣤⣤⣭⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣟⣛⠃⠀⠀⠀⠀⣽⣿⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⡤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⣼⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣿⠛⠁⢀⣠⣤⡀⠀⠀⠀⠀⣤⣤⡶⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠤⠤⢤⣼⣎⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⡼⠿⢿⣿⣿⣦⡀⠀⢼⠿⠃⣀⣤⠤⠄⠀⠀⠀⠀⠀⠀⠀⢀⣴⣶⡿⢾⡿⠊⣱⣦⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⣇⠀⠀⠉⠉⠙⠻⣏⣦⡀⣀⡘⠛⠂⠀⠀⠀⠀⠀⠀⣀⠴⠊⠉⠀⠀⢃⡞⢁⣴⡻⠬⠹⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⡄⠀⠀⠀⠀⢠⣿⣍⣋⣉⡀⠀⠀⠀⠀⣀⡠⠴⠊⠁⠀⠀⠀⠀⠀⡸⠀⠘⡇⢱⠀⢀⢾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⠀⠀⠀⢀⡸⠁⠀⠀⠈⠉⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣔⣁⠜⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⣆⣀⡠⠚⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⡿⠀⠀⠈⠳⣤⡄⠈⠁⠀⠀⠀⠀⢀⣀⣀⣰⣦⠀⠀⠀⠀⠀⠀⠀⣎⡞⠙⡏⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢵⡀⠀⠀⠀⠀⠀⣠⡤⠔⠒⢉⣁⡀⠤⣤⣦⡇⠀⠀⠀⠀⠀⠀⠀⡿⠁⡀⢩⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠷⡄⠀⠀⢰⣮⣁⣴⡲⡽⣋⣀⣀⣿⣟⣛⠃⠀⠀⠀⠀⠀⠀⡠⢃⣴⠁⠸⣿⣿⣿⣿⣿⣽⣿⡿⠿⠽⠟⠛⠻⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡀⠈⢢⡀⠀⠙⣶⠤⢽⡿⠛⠉⠉⢀⣀⠼⠲⡒⠒⠒⠂⢑⡎⠁⢸⠃⢀⣀⡹⠿⠛⠉⠉⠀⠀⠀⠀⠀⠀⡼⣸⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⡇⠀⠀⡷⡄⠀⠈⢻⠻⢤⡴⣲⡚⠉⠀⠀⠀⠁⡀⣀⣰⠟⠀⣠⠗⠚⣉⣤⣶⣤⣴⣤⣴⣦⡥⠀⠀⠀⠁⡇⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠰⣿⣾⣧⣀⡠⠛⠁⠀⠈⢁⡤⣄⡶⢞⣿⠟⠉⣠⣔⣯⣵⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣄⣀⣾⣤⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢫⠃⠀⠀⢰⣿⣿⢋⡡⠀⠀⠀⣠⡴⢫⠟⣩⣶⣿⣥⣾⣟⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢻⠏⠀⠀⢛⡿⠋⠰⠋⠀⠀⢰⡿⠯⠤⠓⠛⠉⣿⠙⣆⢺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⠒⢿⢹⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⠉⠃⠀⠂⠶⠋⠀⡰⢃⣴⠎⠀⠀⠀⠀⠀⠀⠀⣀⣽⣖⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣀⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠘⣱⠟⠁⣠⠴⠂⠀⠶⠖⠚⠉⠁⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣷⣶⣄⣀⣀⠀⠀⠙⠻
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣼⠀⠀⠀⠀⠀⠜⠁⢠⠆⢁⣤⠂⠀⠀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣬⣭⣴⣲
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⣧⠀⠀⠀⠠⠆⢀⢄⡤⠞⠛⠁⣀⢤⠞⠀⠰⠊⠁⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⡿⠛⣏⠄⠀⣀⣤⣴⣿⠀⠀⠀⠀⠀⠈⠈⠀⠀⡰⠛⠉⢀⡄⠀⠀⣀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⡟⢀⣶⣿⣿⡿⣿⣿⣿⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⢠⢆⣈⢤⣠⣺⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
