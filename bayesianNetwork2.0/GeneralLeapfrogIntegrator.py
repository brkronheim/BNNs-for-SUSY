import math

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from inspect import signature

from tensorflow.python.framework.tensor_util import is_tensor

# LeapfrogIntegrator = tfp.python.mcmc.internal
mcmc_util = tfp.python.mcmc.internal.util
class GeneralLeapfrogIntegrator():
  def __init__(self, target_fn, step_sizes, num_steps):
    """Constructs the LeapfrogIntegrator.
    Assumes a simple quadratic kinetic energy function: `0.5 ||momentum||**2`.
    Args:
      target_fn: Python callable which takes an argument like `*state_parts` and
        returns its (possibly unnormalized) log-density under the target
        distribution.
      step_sizes: Python `list` of `Tensor`s representing the step size for the
        leapfrog integrator. Must broadcast with the shape of
        `current_state_parts`.  Larger step sizes lead to faster progress, but
        too-large step sizes make rejection exponentially more likely. When
        possible, it's often helpful to match per-variable step sizes to the
        standard deviations of the target distribution in each variable.
      num_steps: `int` `Tensor` representing  number of steps to run
        the leapfrog integration. Total progress is roughly proportional to
        `step_size * num_steps`.
    """
    self._target_fn = target_fn
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    #self.inputDims = len(signature(self.target_fn).parameters)
    self.Test=False
    self.count=0

  @property
  def target_fn(self):
    return self._target_fn

  @property
  def step_sizes(self):
    return self._step_sizes

  @property
  def num_steps(self):
    return self._num_steps

  def __call__(self,
               momentum_parts,
               state_parts,
               target=None,
               target_grad_parts=None,
               name=None):
    """Applies `num_steps` of the leapfrog integrator.
    Args:
      momentum_parts: Python `list` of `Tensor`s representing momentume for each
        state part.
      state_parts: Python `list` of `Tensor`s which collectively representing
        the state.
      target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `state_parts`.
      target_grad_parts: Python `list` of `Tensor`s representing the gradient of
        `target` with respect to each of `state_parts`.
      name: Python `str` used to group ops created by this function.
    Returns:
      next_momentum_parts: Python `list` of `Tensor`s representing new momentum.
      next_state_parts: Python `list` of `Tensor`s which collectively
        representing the new state.
      next_target: Batch of scalar `Tensor` representing the target (i.e.,
        unnormalized log prob) evaluated at `next_state_parts`.
      next_target_grad_parts: Python `list` of `Tensor`s representing the
        gradient of `next_target` with respect to each of `next_state_parts`.
    """
    with tf.name_scope(name or 'general_leapfrog_integrate'):
      """
      [
          momentum_parts,
          state_parts
      ] = self._process_args(
          momentum_parts,
          state_parts)
      """
      # See Algorithm 1 of "Faster Hamiltonian Monte Carlo by Learning Leapfrog
      # Scale", https://arxiv.org/abs/1810.04449.
      theta=state_parts
      p=momentum_parts
      x=state_parts
      y=momentum_parts

      if (type(theta) != list):
          theta = [theta]
      if (type(p) != list):
          p = [p]
      if (type(x) != list):
          x = [x]
      if (type(y) != list):
          y = [y]
      #print("getting G")
      potential, G, invG, logDet = self.target_fn(*theta)
      #print("G obtained")
      tempP = self.expand(p)
      tempP = self.expand(tempP)
      tempP = tf.reshape(tempP, [1, -1])
      #print("det",det)
      #print("tempP", tempP)
      #print("invG", invG)
      #a=0.5*tf.math.log(det)+0.5*tempP.shape[-1]*tf.math.log((2 * math.pi))

      initial_kinetic = 0.5*logDet+0.5*tempP.shape[-1]*tf.math.log((2 * math.pi)) + tf.matmul(tempP,
                                                                                    tf.matmul(invG,
                                                                                              tf.transpose(tempP))) * 0.5
      #initial_kinetic=c
      initial_kinetic=tf.reshape(initial_kinetic, potential.shape)
      [
          _,
          theta,
          p,
          x,
          y
      ] = tf.while_loop(
          cond=lambda i, *_: i < self.num_steps,
          body=lambda i, *args: [i + 1] + list(self._one_step(*args)),  # pylint: disable=no-value-for-parameter
          loop_vars=[
              tf.zeros_like(self.num_steps, name='iter'),
              theta,
              p,
              x,
              y
          ])

      potential, G, invG, logDet = self.target_fn(*theta)
      tempP = self.expand(p)
      tempP = self.expand(tempP)
      tempP = tf.reshape(tempP, [1, -1])
      final_kinetic = 0.5*logDet+0.5*tempP.shape[-1]*tf.math.log((2 * math.pi)) + tf.matmul(tempP,
                                                                                            tf.matmul(invG,
                                                                                                      tf.transpose(
                                                                                                          tempP))) * 0.5
      final_kinetic = tf.reshape(final_kinetic, potential.shape)
      """
      if len(theta)!= self.inputDims:
          thetaTemp=theta[0]
      else:
          thetaTemp=theta
      """


      return (
          theta,
          initial_kinetic,
          final_kinetic,
          -potential
       )



  def hamiltonian(self, *argv):

     
      if (len(argv) == 1):
          theta = argv[0][:len(argv[0]) // 2]
          p = argv[0][len(argv[0]) // 2:]
      else:
          theta = argv[:len(argv) // 2]
          p = argv[len(argv) // 2:]

      #if len(theta) != self.inputDims:
      #    theta = theta[0]
      #    p = p[0]
      
      potential, G, invG, logDet = self.target_fn(*theta)

      p = self.expand(p)
      p = self.expand(p)
      p = tf.reshape(p, [1, -1])
      #print("det", logDet)
      #print("a",0.5*logDet)
      #print("b", 0.5*p.shape[-1]*tf.math.log((2 * math.pi)))
      #print("p", p)
      #print("mult1", tf.matmul(invG, tf.transpose(p)) * 0.5)
      #print("c",tf.matmul(p,tf.matmul(invG, tf.transpose(p))) * 0.5 )
      kinetic = 0.5*logDet+0.5*p.shape[-1]*tf.math.log((2 * math.pi)) + tf.matmul(p,
                                                                                    tf.matmul(invG, tf.transpose(p))) * 0.5
    
      #print("potential", potential)
      #print("kinetic", kinetic)
      energy = potential + kinetic
      #print("total", energy)
      #print()
      return(energy)

  #@tf.function
  def getGradients(self, theta, p):
      thetaLength=len(theta)
      #print("getGradients")
      #print("list", list(theta)+list(p))
      
      _, grads =mcmc_util.maybe_call_fn_and_grads(self.hamiltonian, list(theta) + list(p))
      #print(grads)
      #print()
      #print()
      if any(g is None for g in grads):
        raise ValueError(
            'Encountered `None` gradient.\n'
            '  state_parts: {}\n'
            '  grads: {}\n'.format(
                theta + p,
                grads))
      delTheta=grads[:thetaLength]
      delP=grads[thetaLength:]
      return(delTheta, delP)

  #@tf.function
  def stepA(self, theta, p, x, y, delta, R):
        #print("p",p)
        delTheta, delY = self.getGradients(theta, y)
        #print()
        #print()
        #print("delTheta", delTheta)
        #print("delY", delY)
        #print()
        #print()
        #for gradient in delTheta:
        #    #print(gradient, -0.5*delta*gradient)
        #print()
        #print("delY")
        #print(delY)
        #print()
        #print("delTheta")
        #print(delTheta)
        deltaP = [-0.5*delta*gradient for gradient in delTheta]
        #print()
        #print("deltaP")
        #print(deltaP)
        deltaX = [0.5*delta*gradient for gradient in delY]
        #print()
        p = [start + update for (start, update) in zip(p, deltaP)]
        x = [start + update for (start, update) in zip(x, deltaX)]

        return(theta, p, x, y)

  #@tf.function
  def stepB(self, theta, p, x, y, delta, R):
        delX, delP = self.getGradients(x, p)

        deltaTheta = [0.5*delta*gradient for gradient in delP]
        deltaY = [-0.5*delta*gradient for gradient in delX]

        theta = [start + update for (start, update) in zip(theta, deltaTheta)]
        y = [start + update for (start, update) in zip(y, deltaY)]

        return(theta, p, x, y)

  #@tf.function
  def stepC(self, theta, p, x, y, delta, R):


        newTheta = [0.5*(T + X + (T-X)*R[0,0] + (P-Y)*R[0,1]) for (T,P,X,Y) in zip(theta, p, x, y)]
        newP = [0.5*(P + Y + (T-X)*R[1,0] + (P-Y)*R[1,1]) for (T,P,X,Y) in zip(theta, p, x, y)]
        newX = [0.5*(T + X - (T-X)*R[0,0] - (P-Y)*R[0,1]) for (T,P,X,Y) in zip(theta, p, x, y)]
        newY = [0.5*(P + Y - (T-X)*R[1,0] - (P-Y)*R[1,1]) for (T,P,X,Y) in zip(theta, p, x, y)]




        theta = newTheta
        p = newP

        x = newX
        y = newY



        return(theta, p, x, y)


  def taoLeapfrogStep(self, theta, p, x, y, delta, R, w, multipleDeltas=False):
        if not multipleDeltas:
            #print("A1",p)
            theta, p, x, y = self.stepA(theta, p, x, y, delta, R)
            #print("after A1", p)
            #print("B1",p)
            theta, p, x, y = self.stepB(theta, p, x, y, delta, R)
            #print("C",p)
            theta, p, x, y = self.stepC(theta, p, x, y, delta, R)
            #print("B2",p)
            theta, p, x, y = self.stepB(theta, p, x, y, delta, R)
            #print("A2",p)
            theta, p, x, y = self.stepA(theta, p, x, y, delta, R)
            #print("End",p)

            return(theta, p, x, y)
        else:
            theta, p, x, y = self.stepA(theta, p, x, y, delta[0], R[0])
            theta, p, x, y = self.stepB(theta, p, x, y, delta[0], R[0])
            theta, p, x, y = self.stepC(theta, p, x, y, delta[0], R[0])
            theta, p, x, y = self.stepB(theta, p, x, y, delta[0], R[0])
            theta, p, x, y = self.stepA(theta, p, x, y, delta[0]+delta[1], R[0])

            for n in range(1,len(delta)-1):

                theta, p, x, y = self.stepB(theta, p, x, y, delta[n], R[n])
                theta, p, x, y = self.stepC(theta, p, x, y, delta[n], R[n])
                theta, p, x, y = self.stepB(theta, p, x, y, delta[n], R[n])
                theta, p, x, y = self.stepA(theta, p, x, y, delta[n]+delta[n+1], R[n])

            theta, p, x, y = self.stepB(theta, p, x, y, delta[-1], R[-1])
            theta, p, x, y = self.stepC(theta, p, x, y, delta[-1], R[-1])
            theta, p, x, y = self.stepB(theta, p, x, y, delta[-1], R[-1])
            theta, p, x, y = self.stepA(theta, p, x, y, delta[-1], R[-1])


            return(theta, p, x, y)

  #@tf.function
  def _one_step(self, theta, p, x, y):

        delta=self._step_sizes[0]
        w=20
        phi = math.pi*(5/4)
        gamma_4=1/(4-4**(1/(4+1)))
        deltas=delta
        #R = np.array([[tf.math.cos(2 * w * delta), tf.math.sin(2 * w * delta)],
        #               [-tf.math.sin(2*w*delta),tf.math.cos(2*w*delta)]])

        R = np.array([[tf.math.cos(phi), tf.math.sin(phi)],
                      [-tf.math.sin(phi), tf.math.cos(phi)]])

        #deltas = [gamma_4*delta,gamma_4*delta,(1-4*gamma_4)*delta,gamma_4*delta,gamma_4*delta]
        #R = [np.array([[tf.math.cos(2*w*deltaNew),tf.math.sin(2*w*deltaNew)],
        #                   [-tf.math.sin(2*w*deltaNew),tf.math.cos(2*w*deltaNew)]]) for deltaNew in deltas]

        #deltas = [gamma_4 * delta, gamma_4 * delta, (1 - 4 * gamma_4) * delta, gamma_4 * delta, gamma_4 * delta]
        #R = [np.array([[tf.math.cos(phi), tf.math.sin(phi)],
        #               [-tf.math.sin(phi), tf.math.cos(phi)]]) for deltaNew in deltas]

        #print("theta before", theta[0:5])
        theta, p, x, y = self.taoLeapfrogStep(theta, p, x, y, deltas, R, w, multipleDeltas=False)
        #print("theta after", theta[0:5])
        return[theta, p, x, y]

  def _process_args(
      self, momentum_parts, state_parts):
      """Sanitize inputs to `__call__`."""
      with tf.name_scope('process_args'):

          momentum_parts = [
              tf.convert_to_tensor(
                  v, dtype_hint=tf.float32, name='momentum_parts')
              for v in momentum_parts]
          state_parts = [
              tf.convert_to_tensor(
                  v, dtype_hint=tf.float32, name='state_parts')
              for v in state_parts]
          return momentum_parts, state_parts

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
