# Graph Neural Networks (GNNs)
1. [An Introduction](https://medium.com/the-modern-scientist/graph-neural-networks-series-part-1-an-introduction-49a88941f888)
2. [Graph Statistics](https://medium.com/the-modern-scientist/graph-neural-networks-series-part-2-graph-statistics-4f271857ec70)
3. [Node embedding](https://medium.com/the-modern-scientist/graph-neural-networks-series-part-3-node-embedding-36613cc967d5)
4. [Message Passing](https://medium.com/the-modern-scientist/graph-neural-networks-series-part-4-the-gnns-message-passing-over-smoothing-e77ffee523cc)
# CS224W
Graph Representation Learning [Text book](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
### CS224W: Machine Learning with Graphs by [Jure Leskovec](https://profiles.stanford.edu/jure-leskovec)
- [Course outline](https://snap.stanford.edu/class/cs224w-2020/)

## What is this course about?
Complex data can be represented as a graph of object relationships. These networks are an important tool for simulating social, technical, and biological systems. This course focuses on the computational, algorithmic, and modeling issues associated with huge network analysis. Students are taught machine learning techniques and data mining tools capable of revealing insights on a range of networks by investigating the underlying graph structure and its properties.

**Topics include:** representation learning and Graph Neural Networks; algorithms for the World Wide Web; reasoning over Knowledge Graphs; influence maximization; disease outbreak detection, social network analysis.

- Mr. Jingbo Yang created a synopsis of this course and made it available to the public. I can't read it now, but I hope it will be beneficial later. Check this out:
  - [CS224W Notes from Fall 2019](https://jingboyang.github.io/cs224w_notes.html)

- Because of their reputation, these two persons are incredibly generous individuals who are uncommon in my school's community. What they've done is quite valuable and has encouraged my own *GNN* studying.
  - [SlidesToCode Repo](https://github.com/mnslarcher/cs224w-slides-to-code)
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
