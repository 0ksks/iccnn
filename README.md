# File Structure

```
\data
    CUB_200_2011.py        # load and process the dataset
    voc2010_crop.py        # load and process the dataset
\network
   \component
      CustomPad2d.py       # customed pad-2d way
      LitNetwork.py        # inherit pl.LightningModule, add conv/relu output hooks
      SMGBlock.py          # similar mask generate block
   \log
      contribution_norm.py # calculate the contribution among different filters
      feature_map.py       # get and save the feature maps of specific layer
   \util
      clustering.py        # perform the clustering
      EMA.py               # exponential moving average on feature map
      normalization.py     # perform the max-min or z-score normalization
      similarity.py        # calculate the similarity based on Pearson correlation
    VGG16BN.py             # implement vgg_16_bn
global_variables.py        # define global varibles
main.py                    # main entry
pretty_print.py            # beautify the output, used during debugging
config.yaml                # config paths
```

**python**: 3.9.19

To run the program, run `python main.py` and follow the instructions provided by the CLI.

run `python main.py --help` to see the parameter definition.