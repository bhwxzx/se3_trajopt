"""Unwrapped wheel spin and touchdown-relative excursion, independently audited."""
import numpy as np
from audit_foot_recovery import hermite_speed_extrema
from nltrajopt.wheel_contact import WheelContactModel


def wheel_motion_metrics(saved,model,csv_q=None,csv_times=None):
    ns=saved['solution']['nodes'];cfg=saved['info']['motion_config']
    times=np.r_[0.,np.cumsum([n['dt'] for n in ns][:-1])]
    q=np.array([n['q'] for n in ns]);v=np.array([n['v'] for n in ns])
    land=cfg['takeoff_time_s']+cfg['flight_duration_s'];touch=int(np.argmin(abs(times-land)))
    wheels=WheelContactModel.from_metadata(model,saved['info']['wheel_contact_model'])
    states=[wheels.evaluate(x) for x in q[touch:]]
    csv_states=[]
    if csv_q is not None and csv_times is not None:
        csv_states=[wheels.evaluate(x) for x,t in zip(csv_q,csv_times) if t>=land-1e-9]
    result={}
    for side,name in zip(('left','right'),wheels.names):
        j=model.joints[model.getJointId(side+'_wheel_joint')]
        speeds=[max(map(abs,hermite_speed_extrema(q[k,j.idx_q],q[k+1,j.idx_q],v[k,j.idx_v],v[k+1,j.idx_v],h))) for k,h in enumerate(np.diff(times))]
        xs=np.array([s[name]['position'][0] for s in states]);delta=xs-xs[0]
        result[side]=dict(terminal_angle_rad=float(q[-1,j.idx_q]),terminal_velocity_rad_s=float(v[-1,j.idx_v]),
            node_peak_speed_rad_s=float(max(abs(v[:,j.idx_v]))),hermite_peak_speed_rad_s=float(max(speeds)),
            speed_limit_rad_s=float(cfg.get('wheel_speed_limit_rad_s',model.velocityLimit[j.idx_v])),
            max_abs_touchdown_relative_x_m=float(max(abs(delta))),
            forward_excursion_m=float(max(delta)),backward_excursion_m=float(-min(delta)),
            net_displacement_m=float(delta[-1]),node_cumulative_path_m=float(sum(abs(np.diff(xs)))))
        result[side]['airborne_node_peak_speed_rad_s']=float(np.max(abs(v[:touch+1,j.idx_v])))
        result[side]['airborne_hermite_peak_speed_rad_s']=float(max(speeds[:touch]))
        result[side]['airborne_speed_limit_rad_s']=cfg.get('airborne_wheel_speed_limit_rad_s')
        if csv_states:
            result[side]['csv_max_abs_touchdown_relative_x_m']=float(max(abs(s[name]['position'][0]-xs[0]) for s in csv_states))
    return result
