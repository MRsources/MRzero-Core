.. _sequence:
.. currentmodule:: MRzeroCore

Sequence
========

A MRI :class:`Sequence` is a Python list of :class:`Repetition` s, each of which is starting with an instantaneous :class:`Pulse` .
The :class:`PulseUsage` of these pulses is only used for automatically building the k-space trajectory for easier reconstruction.
Typical use is similar to the following:

.. code-block:: python

   import MRzeroCore as mr0
   from numpy import pi

   seq = mr0.Sequence()

   # Iterate over the repetitions
   for i in range(64):
       rep = seq.new_rep(2 + 64 + 2)
       
       # Set the pulse
       rep.pulse.usage = mr0.PulseUsage.EXCIT
       rep.pulse.angle = 5 * pi/180
       rep.pulse.phase = i**2 * 117*pi/180 % (2*pi)

       # Set encoding
       rep.gradm[:, :] = ...

       # Set timing
       rep.event_time[:] = ...

       # Set ADC
       rep.adc_usage[2:-2] = 1
       rep.adc_phase[2:-2] = pi - rep.pulse.phase


PulseUsage
----------

.. autoclass:: PulseUsage
    :members:


Pulse
-----

.. autoclass:: Pulse
    :members:


Repetition
----------

.. autoclass:: Repetition
    :members:


Sequence
--------

.. autoclass:: Sequence
    :members:
