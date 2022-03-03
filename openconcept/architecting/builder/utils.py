"""
The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright: (c) 2022, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
         mahmoud.fouda@dlr.de
"""

import numpy as np
from typing import *
import openmdao.api as om
from openmdao.core.component import Component
from openconcept.utilities.dvlabel import DVLabel

__all__ = ['create_output_sum', 'collect_inputs', 'collect_outputs']


def create_output_sum(group: om.Group, output: str, inputs: List[str], units: str = None, n=1) -> Component:
    sum_params = ['sum_input_%d' % i for i in range(len(inputs))]

    sub_name = output+'_sum'
    if len(inputs) == 0:
        comp = group.add_subsystem(
            sub_name, om.IndepVarComp(output, val=np.tile(0., n), units=units), promotes_outputs=[output])
    elif len(inputs) == 1:
        kwargs = {
            output: {'val': np.tile(0., n), 'units': units},
            sum_params[0]: {'val': np.tile(0., n), 'units': units},
        }
        comp = group.add_subsystem(sub_name, om.ExecComp([
            '%s = %s' % (output, sum_params[0]),
        ], **kwargs), promotes_outputs=[output])
    else:
        comp = group.add_subsystem(sub_name, om.AddSubtractComp(output_name=output, input_names=sum_params, units=units,
                                                                length=n), promotes_outputs=[output])

    for i, param in enumerate(sum_params):
        group.connect(inputs[i], sub_name+'.'+param)

    return comp


def collect_inputs(group: om.Group, inputs: List[Tuple[str, Optional[str], Optional[Union[float, List[float]]]]],
                   name: str = None) -> Tuple[DVLabel, Dict[str, str]]:
    label_map = {inp[0]: inp[0]+'_pass' for inp in inputs}

    comp = group.add_subsystem(
        name or 'in_collect', DVLabel([[inp, label_map[inp], val, units] for inp, units, val in inputs]),
        promotes_inputs=['*'], promotes_outputs=['*'])

    return comp, label_map


def collect_outputs(group: om.Group, outputs: List[Tuple[str, Optional[str]]], name: str = None) \
        -> Tuple[DVLabel, Dict[str, str]]:
    label_map = {out: out+'_pass' for out, _ in outputs}

    comp = group.add_subsystem(
        name or 'out_collect', DVLabel([[label_map[out], out, 1., units] for out, units in outputs]),
        promotes_inputs=['*'], promotes_outputs=['*'])

    return comp, label_map
