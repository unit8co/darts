# Frequently Asked Questions

- *Is Darts just a wrapper around other libraries?*

  No. When it makes sense, we reuse existing implementations (e.g. from statsforecasts), but we often write our 
  own implementations (e.g., of neural networks). Furthermore, Darts models often have more features than their original counter-parts. For instance, unlike the original version, our implementation of N-BEATS supports multivariate time series, past covariates, and probabilistic forecasts.

- *Darts looks like an awesome project, can I contribute?*

  Absolutely! We are constantly welcoming contributions from the community. If you contribute, you will be acknowledged in the [wall of fame (a.k.a the changelog)](https://github.com/unit8co/darts/blob/master/CHANGELOG.md)! Contributions don't have to be code only but can also be e.g., documentation. In addition, we're also happy to receive suggestions in the form of issues on Github. The best place to start for contributors is the [contribution guidelines](https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md).

- *I have some ideas for contributing a new model into Darts, is that possible?*

  In general, yes, we do welcome reference implementations of new models. However, we are loosely filtering to keep models that are either classics, or convincingly shown (e.g., in a paper or some other forms of evidence) to be state-of-the-art in some respect.

- *How can I make Darts work on Google Colab?*

  Colab may have issues with recent versions of pyyaml. Installing pyyaml 5.4.1 before installing Darts solves these issues:
  ```
  !pip install pyyaml==5.4.1
  ```

- *I'm getting some NaNs in my forecasts, what should I do?*

  Usually this means one of two things:
  - You have some NaNs in your training data (targets or covariates). This is the most frequent situation, and it will lead most models
    to always forecast NaNs.
  - Training the model causes some numerical divergence. If you're working with neural networks, make sure your data is properly scaled, and if the issue
    persists, try reducing the learning rate.

- *My forecasting models give bad results, can you help?*

  Getting good forecasts is about more than just calling the `fit()`/`predict()` functions, and always involves some data science work to understand which approaches are appropriate. We cannot give general answers, however if you have important forecasting problems or need help to industrialize your forecasts, Unit8 provides technical consulting. <a href="mailto:info@unit8.co">Feel free to contact us</a>.
