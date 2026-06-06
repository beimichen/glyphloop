"""Image embedding + clustering — the engine of the cold-start sequel.

Embed unlabeled images, cluster them, and let the clusters decide which question
to ask a human (see :mod:`glyphloop.active`). A TensorFlow-free pixel embedding is
provided so the clustering machinery (and the committed figure) runs without
weights; a learned encoder embedding is available with the ``[train]`` extra.
"""
