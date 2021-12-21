
"""Collection of all defined agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rl_planning import mopp_bc_agent, mopp_agent, mbop_agent

AGENT_MODULES_DICT = {
    # planing:
    'mopp_bc': mopp_bc_agent,
    'mopp': mopp_agent,
    'mbop': mbop_agent,
}
