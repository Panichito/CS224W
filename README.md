# CS224W
### CS224W: Machine Learning with Graphs by [Jure Leskovec](https://profiles.stanford.edu/jure-leskovec)
- [Course outline](http://snap.stanford.edu/class/cs224w-2021/)

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
