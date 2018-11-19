Model
===========================

We consider a stochastic dynamical system of :math:`N` variables in
which the state at time :math:`t+1`, :math:`\vec{\sigma} (t+1)` is
governed by the state at time :math:`t`, :math:`\vec{\sigma} (t)`,
according to the kinetic Ising model

.. math::

   \label{eq:kIsingProb}
   P[\sigma_i(t+1)|\vec{\sigma}(t)] = \frac{\exp [ \sigma_i(t+1) H_i(t)]}{\mathcal{N}}

where :math:`H_i(t) = \sum_j W_{ij} \sigma_j(t)` represents the local
field acting on :math:`\sigma_i(t+1)`, :math:`W_{ij}` represents the
interaction from variable :math:`j` to variable :math:`i`, and
:math:`\mathcal{N} = \sum_{\sigma_i(t+1)} \exp[\sigma_i(t+1) H_i(t)]`
represents a normalizing factor. Intuitively, the state
:math:`\sigma_i(t+1)` tends to align with the local field
:math:`H_i(t)`.

| The inference methods in the literature usually extract the
  interactions :math:`W_{ij}` from the time series
  :math:`\{ \vec{\sigma}(t) \}` of entire variables. In realistic
  situations, however, the experimental data often contains only subset
  of variables. Our aim was to develop a data driven approach that can
  infer
| (i) the interactions between variables (including
  observed-to-observed, hidden-to-observed, observed-to-hidden, and
  hidden-to-hidden interactions);
| (ii) the configurations of hidden variables;
| (iii) the number of hidden variables.
