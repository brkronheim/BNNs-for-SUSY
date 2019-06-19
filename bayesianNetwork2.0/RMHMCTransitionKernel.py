
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow_probability.python import distributions
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

from inspect import signature

from GeneralLeapfrogIntegrator import GeneralLeapfrogIntegrator

UncalibratedRiemannManifoldHamiltonianMonteCarloKernelResults = collections.namedtuple(
    'UncalibratedRiemannManifoldHamiltonianMonteCarloKernelResults',
    [
        'log_acceptance_correction',
        'target_log_prob',        # For "next_state".
        'step_size',
        'num_leapfrog_steps',
    ])

RiemannManifoldHamiltonianMonteCarloExtraKernelResults = collections.namedtuple(
    'HamiltonianMonteCarloExtraKernelResults',
    [
        'step_size_assign',
    ])

class RiemannManifoldHamiltonianMonteCarlo(kernel_base.TransitionKernel):
  
  @deprecation.deprecated_args(
      '2019-05-22', 'The `step_size_update_fn` argument is deprecated. Use '
      '`tfp.mcmc.SimpleStepSizeAdaptation` instead.', 'step_size_update_fn')
  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               step_size_update_fn=None,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      step_size_update_fn: Python `callable` taking current `step_size`
        (typically a `tf.Variable`) and `kernel_results` (typically
        `collections.namedtuple`) and returns updated step_size (`Tensor`s).
        Default value: `None` (i.e., do not update `step_size` automatically).
      seed: Python integer to seed the random number generator.
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly. This is incompatible with `step_size_update_fn`,
        which must be set to `None`.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if step_size_update_fn and store_parameters_in_results:
      raise ValueError('It is invalid to simultaneously specify '
                       '`step_size_update_fn` and set '
                       '`store_parameters_in_results` to `True`.')

    impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedRiemannManifoldHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            state_gradients_are_stopped=state_gradients_are_stopped,
            seed=seed,
            name='rm_hmc_kernel' if name is None else name,
            store_parameters_in_results=store_parameters_in_results),
        seed=seed)
    parameters = impl.inner_kernel.parameters.copy()

    parameters['step_size_update_fn'] = step_size_update_fn
    self._impl = impl
    self._parameters = parameters


  @property
  def target_log_prob_fn(self):
    return self._impl.inner_kernel.target_log_prob_fn

  @property
  def step_size(self):
    """Returns the step_size parameter.
    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `step_size` placed in the kernel
    results by the `bootstrap_results` method. The actual step size in that
    situation is governed by the `previous_kernel_results` argument to
    `one_step` method.
    Returns:
      step_size: A floating point `Tensor` or a list of such `Tensors`.
    """
    return self._impl.inner_kernel.step_size

  @property
  def num_leapfrog_steps(self):
    """Returns the num_leapfrog_steps parameter.
    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `num_leapfrog_steps` placed in
    the kernel results by the `bootstrap_results` method. The actual
    `num_leapfrog_steps` in that situation is governed by the
    `previous_kernel_results` argument to `one_step` method.
    Returns:
      num_leapfrog_steps: An integer `Tensor`.
    """
    return self._impl.inner_kernel.num_leapfrog_steps

  @property
  def state_gradients_are_stopped(self):
    return self._impl.inner_kernel.state_gradients_are_stopped

  @property
  def step_size_update_fn(self):
    return self._parameters['step_size_update_fn']

  @property
  def seed(self):
    return self._impl.inner_kernel.seed

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  #@tf.function
  def one_step(self, current_state, previous_kernel_results):
    """Runs one iteration of Hamiltonian Monte Carlo.
    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
    """
    previous_step_size_assign = (
        [] if self.step_size_update_fn is None
        else (previous_kernel_results.extra.step_size_assign
              if mcmc_util.is_list_like(
                  previous_kernel_results.extra.step_size_assign)
              else [previous_kernel_results.extra.step_size_assign]))

    with tf.control_dependencies(previous_step_size_assign):
      next_state, kernel_results = self._impl.one_step(
          current_state, previous_kernel_results)
      if self.step_size_update_fn is not None:
        step_size_assign = self.step_size_update_fn(  # pylint: disable=not-callable
            self.step_size, kernel_results)
        kernel_results = kernel_results._replace(
            extra=RiemannManifoldHamiltonianMonteCarloExtraKernelResults(
                step_size_assign=step_size_assign))

    return next_state, kernel_results

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    kernel_results = self._impl.bootstrap_results(init_state)
    if self.step_size_update_fn is not None:
      step_size_assign = self.step_size_update_fn(self.step_size, None)  # pylint: disable=not-callable
      kernel_results = kernel_results._replace(
          extra=RiemannManifoldHamiltonianMonteCarloExtraKernelResults(
              step_size_assign=step_size_assign))
    return kernel_results


class UncalibratedRiemannManifoldHamiltonianMonteCarlo(kernel_base.TransitionKernel):
  """Runs one step of Uncalibrated Hamiltonian Monte Carlo.
  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use `HamiltonianMonteCarlo(...)`
  or `MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...))`.
  For more details on `UncalibratedHamiltonianMonteCarlo`, see
  `HamiltonianMonteCarlo`.
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: `Tensor` or Python `list` of `Tensor`s representing the step
        size for the leapfrog integrator. Must broadcast with the shape of
        `current_state`. Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
        for. Total progress per HMC step is roughly proportional to
        `step_size * num_leapfrog_steps`.
      state_gradients_are_stopped: Python `bool` indicating that the proposed
        new state be run through `tf.stop_gradient`. This is particularly useful
        when combining optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed: Python integer to seed the random number generator.
      store_parameters_in_results: If `True`, then `step_size` and
        `num_leapfrog_steps` are written to and read from eponymous fields in
        the kernel results objects returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if seed is not None and tf.executing_eagerly():
      # TODO(b/68017812): Re-enable once TFE supports `tf.random_shuffle` seed.
      raise NotImplementedError('Specifying a `seed` when running eagerly is '
                                'not currently supported. To run in Eager '
                                'mode with a seed, use `tf.set_random_seed`.')
    self._seed_stream = distributions.SeedStream(seed, 'hmc_one_step')
    if not store_parameters_in_results:
      mcmc_util.warn_if_parameters_are_not_simple_tensors(
          dict(step_size=step_size, num_leapfrog_steps=num_leapfrog_steps))
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        state_gradients_are_stopped=state_gradients_are_stopped,
        seed=seed,
        name=name,
        store_parameters_in_results=store_parameters_in_results,
    )
    self._momentum_dtype = None

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    """Returns the step_size parameter.
    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `step_size` placed in the kernel
    results by the `bootstrap_results` method. The actual step size in that
    situation is governed by the `previous_kernel_results` argument to
    `one_step` method.
    Returns:
      step_size: A floating point `Tensor` or a list of such `Tensors`.
    """
    return self._parameters['step_size']

  @property
  def num_leapfrog_steps(self):
    """Returns the num_leapfrog_steps parameter.
    If `store_parameters_in_results` argument to the initializer was set to
    `True`, this only returns the value of the `num_leapfrog_steps` placed in
    the kernel results by the `bootstrap_results` method. The actual
    `num_leapfrog_steps` in that situation is governed by the
    `previous_kernel_results` argument to `one_step` method.
    Returns:
      num_leapfrog_steps: An integer `Tensor`.
    """
    return self._parameters['num_leapfrog_steps']

  @property
  def state_gradients_are_stopped(self):
    return self._parameters['state_gradients_are_stopped']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @property
  def _store_parameters_in_results(self):
    return self._parameters['store_parameters_in_results']

  def expand(self, current):
    """Expands tensors to that they are of rank 2

    Arguments:
        * current: tensor to expand
    Returns:
        * expanded: expanded tensor

    """
    currentShape=tf.pad(
            tf.shape(current),
            paddings=[[tf.where(tf.rank(current) > 1, 0, 1), 0]],
            constant_values=1)
    expanded=tf.reshape(current, currentShape)
    return(expanded)
  
  #@tf.function
  def get_hessian(self, argv):
          n = None
          gradeint=None
          _gradients=None
          #G = None
          tensorList = []
          for x in range(len(argv)):
              tensorList.append(argv[x])
          with tf.GradientTape(persistent=True, watch_accessed_variables=False) as SecondTape:
              for x in tensorList:     
                  SecondTape.watch(x)
              with tf.GradientTape(persistent=False, watch_accessed_variables=False) as FirstTape:
                  FirstTape.watch(tensorList)
                  argv = tensorList
                  theta = []
                  index = 0
                  for info in self.restoreShapes:
                      theta.append(tf.reshape(argv[index:index + info[1]], info[0]))
                      index += info[1]
                  potential = self.target_log_prob_fn(*theta)
              _gradients = FirstTape.gradient(potential, tensorList, unconnected_gradients=tf.UnconnectedGradients.NONE)
              SecondTape.watch(_gradients)
              #gradient=tf.convert_to_tensor(_gradients)
              gradient=tf.stack(_gradients)
              #n = array_ops.size(tensorList)
              n=len(tensorList)
          """
          loop_vars = [
                  array_ops.constant(0, tf.int32),
                  tensor_array_ops.TensorArray(tf.float32, n)
          ]

          gradientIter=iter(_gradients)
          _, hessian = control_flow_ops.while_loop(
              lambda j, _: j < n,
              lambda j, result: (j + 1,
                         result.write(j, SecondTape.gradient(next(gradientIter), tensorList, unconnected_gradients=tf.UnconnectedGradients.ZERO))),
              loop_vars
          )
          """
        
          hessian = SecondTape.jacobian(gradient, tensorList, unconnected_gradients=tf.UnconnectedGradients.ZERO)
          #_shape = array_ops.shape(tensorList)
          #_reshaped_hessian = array_ops.reshape(hessian.stack(),
          #                                         array_ops.concat((_shape, _shape), 0))
          #hessians = _reshaped_hessian
          #G = -hessians
          #print(hessian)
          G=-tf.convert_to_tensor(hessian)
          #print(G)
        
          """
          loop_vars = [
                array_ops.constant(0, tf.int32),
                tensor_array_ops.TensorArray(tf.float32, n)
            ]
          #print(gradient)
          #print(n)
          for x in range(n):
                #print(gradient[x])
          #print(gradient[0])
          #print(gradient[1])
          _, hessian = control_flow_ops.while_loop(
                lambda j, _: j < n,
                lambda j, result: (j + 1,
                                   result.write(j, SecondTape.gradient(gradient[j], tensorList, unconnected_gradients=tf.UnconnectedGradients.ZERO))),
                loop_vars
          )
        
          _shape = array_ops.shape(tensorList)
          _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                                   array_ops.concat((_shape, _shape), 0))
          hessians = _reshaped_hessian
          G = -hessians
          """
            
            
            
          # Compute first-order derivatives and iterate for each x in xs.
          """
          hessians = []
          _gradients = gradients(ys, xs, **kwargs)
          for gradient, x in zip(_gradients, xs):
            # change shape to one-dimension without graph branching
            gradient = array_ops.reshape(gradient, [-1])

            # Declare an iterator and tensor array loop variables for the gradients.
            
            # Iterate over all elements of the gradient and compute second order
            # derivatives.
          """
        
        
        
        
          
          return(G, potential)

  #@tf.function          
  def hamiltonian_energy_fn(self, *argv):
      #print()
      G,potential=self.get_hessian(argv)
      #print("potential", potential)
      #print()
      #print(G)
      #smallG=tf.minimum(G, 1e6)
      #smallG=tf.maximum(smallG, -1e6)  

      #condG=tf.reduce_any(tf.math.not_equal(G, smallG))
      #G = smallG
      s, u, v = tf.linalg.svd(G)
      safeInvS=1/(tf.where(tf.reduce_any(tf.math.equal(s,0)), tf.ones_like(s),s))
      safeLog=tf.math.log(tf.where(tf.reduce_any(tf.math.less_equal(s, 0)), tf.ones_like(s), s))
      condA=tf.reduce_any(s==0)
      #condB=tf.reduce_any(tf.math.not_equal(G, smallG))
      #logDet = tf.where(tf.reduce_any([condA, condB, condG]),tf.constant(0.0),tf.reduce_sum(safeLog))
      logDet = tf.where(tf.reduce_any([condA]),tf.constant(0.0),tf.reduce_sum(safeLog))
      
      #invG=tf.where(tf.reduce_any([condA, condB, condG]), tf.linalg.diag(tf.ones_like(s)), 
      #               tf.matmul(v ,tf.matmul(tf.linalg.diag(safeInvS), tf.transpose(u))))
        
      invG=tf.where(tf.reduce_any([condA]), tf.linalg.diag(tf.ones_like(s)), 
                     tf.matmul(v ,tf.matmul(tf.linalg.diag(safeInvS), tf.transpose(u))))
      #print(-potential)
      #print(G)
      #print(invG)
      #print(logDet)
      #print()
      return(-potential, G, invG, logDet)
  
  #@tf.function
  def run_integrator(self,step_sizes, num_leapfrog_steps,current_momentum_parts, current_state_parts):
      integrator = GeneralLeapfrogIntegrator(
          self.hamiltonian_energy_fn, step_sizes, num_leapfrog_steps)
      [
          next_state_parts,
          initial_kinetic,
          final_kinetic,
          final_target_log_prob
      ] = integrator(current_momentum_parts,
                     current_state_parts)
      print("ik", initial_kinetic)
      print("fk", final_kinetic)
      return(next_state_parts,
          initial_kinetic,
          final_kinetic,
          final_target_log_prob)
    
  @mcmc_util.set_doc(RiemannManifoldHamiltonianMonteCarlo.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results):
    with tf.compat.v2.name_scope(
        mcmc_util.make_name(self.name, 'hmc', 'one_step')):
      if self._store_parameters_in_results:
        step_size = previous_kernel_results.step_size
        num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
      else:
        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps
      [
          current_state_parts,
          step_sizes,
          current_target_log_prob,
      ] = _prepare_args(
          self.target_log_prob_fn,
          current_state,
          step_size,
          previous_kernel_results.target_log_prob,
          maybe_expand=True,
          state_gradients_are_stopped=self.state_gradients_are_stopped)

      self.restoreShapes = []
      for x in current_state_parts:
          n = 1
          shape = x.shape
          for m in shape:
              n *= m
          self.restoreShapes.append([shape, n])
      current_state_parts = [tf.reshape(part, [-1]) for part in current_state_parts]
      current_state_parts = tf.concat(current_state_parts, -1)
      temp=[]
      #print(current_state_parts)
      for x in range(current_state_parts.shape[0]):
            temp.append(current_state_parts[x])
      current_state_parts=temp
      #print(current_state_parts)
    
        
      current_momentum_parts = []

      for x in current_state_parts:
          current_momentum_parts.append(tf.random.normal(
                shape=tf.shape(input=x),
                dtype=self._momentum_dtype or x.dtype.base_dtype,
                seed=self._seed_stream()))


      next_state_parts, initial_kinetic, final_kinetic, final_target_log_prob =self.run_integrator(step_sizes, num_leapfrog_steps,current_momentum_parts, current_state_parts)
    
    
      if self.state_gradients_are_stopped:
        next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      independent_chain_ndims = distribution_util.prefer_static_rank(
          current_target_log_prob)
      
      next_state_parts = maybe_flatten(next_state_parts)

      new_kernel_results = previous_kernel_results._replace(
          log_acceptance_correction=_compute_log_acceptance_correction(
              initial_kinetic, final_kinetic,
              independent_chain_ndims),
          target_log_prob=final_target_log_prob
      )
      argv = next_state_parts#[0]
      next_state_parts = []
      index = 0
      #print(self.restoreShapes)
      for info in self.restoreShapes:
          next_state_parts.append(tf.reshape(argv[index:index + info[1]], info[0]))
          index += info[1]

      return next_state_parts, new_kernel_results

  @mcmc_util.set_doc(RiemannManifoldHamiltonianMonteCarlo.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
      with tf.compat.v2.name_scope(
              mcmc_util.make_name(self.name, 'hmc', 'bootstrap_results')):
          if not mcmc_util.is_list_like(init_state):
              init_state = [init_state]
          if self.state_gradients_are_stopped:
              init_state = [tf.stop_gradient(x) for x in init_state]
          else:
              init_state = [tf.convert_to_tensor(value=x) for x in init_state]
          [
              init_target_log_prob,
              init_grads_target_log_prob,
          ] = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn, init_state)
          if self._store_parameters_in_results:
              return UncalibratedRiemannManifoldHamiltonianMonteCarloKernelResults(
                  log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                  target_log_prob=init_target_log_prob,
                  step_size=tf.nest.map_structure(
                      lambda x: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
                          value=x,
                          dtype=init_target_log_prob.dtype,
                          name='step_size'),
                      self.step_size),
                  num_leapfrog_steps=tf.convert_to_tensor(
                      value=self.num_leapfrog_steps,
                      dtype=tf.int32,
                      name='num_leapfrog_steps'))
          else:
              return UncalibratedRiemannManifoldHamiltonianMonteCarloKernelResults(
                  log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                  target_log_prob=init_target_log_prob,
                  step_size=[],
                  num_leapfrog_steps=[]
              )


def _compute_log_acceptance_correction(current_kinetic,
                                       proposed_kinetic,
                                       independent_chain_ndims,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.
  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:
  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```
  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.
  Inserting this into the detailed balance equation implies:
  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```
  One definition of `a(x'|x)` which satisfies (*) is:
  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```
  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)
  We call the bracketed term the "acceptance correction".
  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Assuming a standard Gaussian distribution for momentums, the chain eventually
  converges to:
  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  ```
  Relating this back to Metropolis-Hastings parlance, for HMC we have:
  ```none
  p([x, z]) propto= target_prob(x) exp(-0.5 z**2)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```
  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:
  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [exp(-0.5 z**2) / exp(-0.5 z'**2)]
                       target_prob(x)
  ```
  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)
  Args:
    current_momentums: `Tensor` representing the value(s) of the current
      momentum(s) of the state (parts).
    proposed_momentums: `Tensor` representing the value(s) of the proposed
      momentum(s) of the state (parts).
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leftmost `Tensor` dimensions which index independent chains.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').
  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.compat.v2.name_scope(name or 'compute_log_acceptance_correction'):
      """
      #current_momentums=tf.reshape(current_momentums,proposed_momentums.shape)
      log_current_kinetic, log_proposed_kinetic = [], []
      for current_momentum, proposed_momentum in zip(
          current_momentums, proposed_momentums):
        axis = tf.range(independent_chain_ndims, tf.rank(current_momentum))
        log_current_kinetic.append(_log_sum_sq(current_momentum, axis))
        log_proposed_kinetic.append(_log_sum_sq(proposed_momentum, axis))
      current_kinetic = 0.5 * tf.exp(
          tf.reduce_logsumexp(
              input_tensor=tf.stack(log_current_kinetic, axis=-1), axis=-1))
      proposed_kinetic = 0.5 * tf.exp(
          tf.reduce_logsumexp(
              input_tensor=tf.stack(log_proposed_kinetic, axis=-1), axis=-1))
      """
      return mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])#*0.0


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [
      tf.convert_to_tensor(value=s, name='current_state') for s in state_parts
  ]
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, _ = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      None)
  step_sizes = (list(step_size) if mcmc_util.is_list_like(step_size)
                else [step_size])
  step_sizes = [
      tf.convert_to_tensor(
          value=s, name='step_size', dtype=target_log_prob.dtype)
      for s in step_sizes
  ]
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob
  ]


def _log_sum_sq(x, axis=None):
  """Computes log(sum(x**2))."""
  return tf.reduce_logsumexp(
      input_tensor=2. * tf.math.log(tf.abs(x)), axis=axis)
