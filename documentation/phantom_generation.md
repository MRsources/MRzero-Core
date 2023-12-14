(generating_phantoms)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Generating Phantoms

Phantoms are built from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) data. This data is not included directly.
Instead, a BrainWeb downloader is part of `mr0`, that can be run once to download all segmentation data provided by BrainWeb, which is then filled to produce files that can be loaded as mentioned [here](load_brainweb).

Phantoms can be generated with different configurations, depending on the use-case, like 3T or 7T data, high-res, or whether to include fat are options. A fixed set of configuartions facilitates reproducibility. To execute generation, just run the following code:

```python
import MRzeroCore as mr0

mr0.generate_brainweb_phantoms("output/brainweb", "3T")
```

```{eval-rst}
.. autofunction:: MRzeroCore.phantom.brainweb.generate_brainweb_phantoms
```
