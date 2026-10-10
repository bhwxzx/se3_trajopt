"""Independent physical and motion-intent checks for a saved motion candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from check_leg_to_wheel_dynamics import URDF,analyze,csv_error
from lw_validation_helpers import write_json
from visualize_trajectory_csv import load_configurations


def validate(solution,csv_path,output):
    from nltrajopt.wheel_contact import WheelContactModel
    saved=json.loads(solution.read_text());nodes=saved['solution']['nodes'];config=saved['info']['motion_config']
    model=pin.buildModelFromUrdf(str(URDF),pin.JointModelFreeFlyer());data=model.createData()
    wheels=WheelContactModel.from_metadata(model,saved['info']['wheel_contact_model']);mass=pin.computeTotalMass(model)
    times=np.r_[0.,np.cumsum([n['dt'] for n in nodes][:-1])]
    takeoff=next(k for k,n in enumerate(nodes) if not n['forces'])
    q0=np.array(nodes[0]['q']);qend=np.array(nodes[-1]['q'])
    initial_com=pin.centerOfMass(model,data,q0).copy()
    pin.forwardKinematics(model,data,q0);pin.updateFramePlacements(model,data)
    foot_refs={name:data.oMf[model.getFrameId(name)].translation.copy() for name in nodes[0]['forces']}
    normal_acc=[];full_acc=[];support_wheel_x=[];dependent_sv=[]
    foot_errors=[];newton=[];friction=[];hip=[];poses=[];coms=[];vs=[];momenta=[];wheel_speed=[];wheel_height=[]
    stance_speeds=[];stance_acc=[];stance_knees=[]
    for k,n in enumerate(nodes):
        q,v,a=[np.array(n[k]) for k in ('q','v','a')]
        pin.forwardKinematics(model,data,q,v,a);pin.updateFramePlacements(model,data)
        forces=[data.oMf[model.getFrameId(name)].rotation@np.array(f) for name,f in n['forces'].items()]
        for name in foot_refs:
            if name in n['forces'] or (config.get('fixed_foot_integration') and k==takeoff):
                foot_errors.append(np.linalg.norm(data.oMf[model.getFrameId(name)].translation-foot_refs[name],np.inf))
                stance_speeds.append(np.linalg.norm(pin.getFrameVelocity(model,data,model.getFrameId(name),pin.LOCAL).vector,np.inf))
                if k<takeoff:
                    stance_acc.append(np.linalg.norm(pin.getFrameAcceleration(model,data,model.getFrameId(name),pin.LOCAL).vector,np.inf))
        if k<=takeoff:
            stance_knees.extend([data.oMi[model.getJointId(side+'_shank_joint')].translation[2] for side in ('left','right')])
        com=pin.centerOfMass(model,data,q,v,a).copy();coms.append(com);vs.append(data.vcom[0].copy())
        newton.append(np.max(abs(np.sum(forces,axis=0)+mass*model.gravity.linear-mass*data.acom[0])))
        momenta.append(pin.computeCentroidalMomentum(model,data,q,v).angular.copy())
        for f in forces:friction.extend([.9*f[2]-abs(f[0]),.9*f[2]-abs(f[1]),f[2],1.5*mass*np.linalg.norm(model.gravity.linear)-np.linalg.norm(f)])
        ws=wheels.evaluate(q,v,derivatives=True);poses.append(np.mean([s['position'] for s in ws.values()],axis=0))
        if all(name in n['forces'] for name in wheels.names):
            support_wheel_x.extend([float(state['position'][0]) for state in ws.values()])
            full_acc.extend([float(np.max(abs(state['velocity_dq']@v+state['Jmaterial']@a))) for state in ws.values()])
            normal_acc.extend([abs(state['velocity_dq'][2]@v+state['Jmaterial'][2]@a) for state in ws.values()])
            matrix=np.array([state['Jgeometry'][2,[2,3]] for state in ws.values()])
            dependent_sv.append(np.linalg.svd(matrix,compute_uv=False)[-1])
        for name,state in ws.items():
            if name in n['forces']:
                wheel_speed.append(np.linalg.norm(state['velocity'],np.inf));wheel_height.append(abs(state['position'][2]))
        for side in ('left','right'):
            j=model.joints[model.getJointId(side+'_hip_joint')];hip.append([q[j.idx_q],v[j.idx_v],a[j.idx_v]])
    coms=np.array(coms);vs=np.array(vs);poses=np.array(poses);momenta=np.array(momenta)
    takeoff=next(k for k,n in enumerate(nodes) if not n['forces']);land=next(k for k,n in enumerate(nodes) if all(name in n['forces'] for name in wheels.names))
    dt=times[takeoff:land+1]-times[takeoff]
    velocity_error=vs[takeoff:land+1]-vs[takeoff]-dt[:,None]*model.gravity.linear
    position_error=coms[takeoff:land+1]-coms[takeoff]-dt[:,None]*vs[takeoff]-.5*dt[:,None]**2*model.gravity.linear
    momentum_error=momenta[takeoff:land+1]-momenta[takeoff]
    terminal_mid=np.mean([s['position'] for s in wheels.evaluate(qend).values()],axis=0)
    csv_q=load_configurations(csv_path,model)
    csv_hip=max(abs(q[model.joints[model.getJointId(side+'_hip_joint')].idx_q]) for q in csv_q for side in ('left','right'))
    apex=coms[takeoff:land+1,2].max();report,_,torque,_,_=analyze(saved,model)
    exporter_check=csv_error(nodes,model,csv_path)
    metadata_file=csv_path.parent/'export_metadata.json'
    if metadata_file.exists():
        from export_leg_to_wheel_motion import reconstruction_error,METHOD,CONTACT_METHOD
        if json.loads(metadata_file.read_text())['method'] not in (METHOD,CONTACT_METHOD):raise ValueError('Unrecognized export method')
        exporter_check=reconstruction_error(nodes,model,csv_path)
    metrics={'max_hip_q_v_a_abs':np.max(abs(np.array(hip)),axis=0).tolist(),'max_csv_hip_angle_abs_rad':float(csv_hip),
             'flight_com_apex_above_initial_m':float(apex-initial_com[2]),
             'touchdown_com_minus_wheel_x_m':float(coms[land,0]-poses[land,0]),
             'touchdown_wheel_minus_target_x_m':float(poses[land,0]-terminal_mid[0]),
             'max_all_original_sole_point_position_error_m':float(max(foot_errors)),
             'max_COM_Newton_residual_N':float(max(newton)),
             'min_world_friction_force_margin_N':float(min(friction)),
             'max_flight_ballistic_position_error_m':float(np.max(abs(position_error))),
             'max_flight_ballistic_velocity_error_m_s':float(np.max(abs(velocity_error))),
             'max_flight_angular_momentum_error_kg_m2_s':float(np.max(abs(momentum_error))),
             'max_active_wheel_material_velocity_m_s':float(max(wheel_speed)),
             'max_active_wheel_geometry_height_m':float(max(wheel_height)),
             'max_joint_torque_Nm':float(np.max(abs(torque))),
             'max_joint_torque_limit_fraction':max(j['limit_fraction'] for j in report['joint_torques']),
             'rnea_vs_virtual_work_max_abs':report['rnea_vs_jacobian_max_abs'],
             'csv_export':exporter_check,
             'max_wheel_normal_acceleration_m_s2':float(max(normal_acc)),
             'min_dependent_contact_jacobian_singular_value':float(min(dependent_sv))}
    tol=1e-6
    checks={'hips_fixed_zero':max(metrics['max_hip_q_v_a_abs'])<1e-10 and csv_hip<1e-10,
            'jump_height':metrics['flight_com_apex_above_initial_m']>=config['apex_above_initial_com_m']-tol,
            'touchdown_alignment':abs(metrics['touchdown_com_minus_wheel_x_m'])<=config['touchdown_com_wheel_x_tolerance_m']+tol,
            'touchdown_position':abs(metrics['touchdown_wheel_minus_target_x_m'])<=config['touchdown_target_x_tolerance_m']+tol,
            'original_sole_points_fixed':metrics['max_all_original_sole_point_position_error_m']<tol,
            'COM_Newton':metrics['max_COM_Newton_residual_N']<tol,
            'friction_and_force_bounds':metrics['min_world_friction_force_margin_N']>=-tol,
            'ballistic_position':metrics['max_flight_ballistic_position_error_m']<tol,
            'ballistic_velocity':metrics['max_flight_ballistic_velocity_error_m_s']<tol,
            'flight_angular_momentum':metrics['max_flight_angular_momentum_error_kg_m2_s']<tol,
            'wheel_material_velocity':metrics['max_active_wheel_material_velocity_m_s']<tol,
            'wheel_geometry':metrics['max_active_wheel_geometry_height_m']<tol,
            'torque_hardware_bounds':metrics['max_joint_torque_limit_fraction']<=1.+tol,
            'export_matches_saved_nodes':metrics['csv_export']['matches']}
    if config.get('projected_wheel_support'):
        checks['normal_contact_acceleration']=metrics['max_wheel_normal_acceleration_m_s2']<tol
        if config.get('wheel_support_integrator') in ('full_rolling','sagittal_rolling','coaxial_rolling'):
            metrics['max_wheel_full_acceleration_m_s2']=max(full_acc)
            checks['full_rolling_contact_acceleration']=max(full_acc)<tol
        checks['dependent_contact_coordinates_nonsingular']=metrics['min_dependent_contact_jacobian_singular_value']>1e-4
    if config.get('fixed_foot_integration'):
        metrics['max_stance_and_takeoff_foot_spatial_velocity']=float(max(stance_speeds))
        metrics['max_stance_foot_spatial_acceleration']=float(max(stance_acc))
        checks['stance_and_takeoff_no_slip']=max(stance_speeds)<tol
        checks['stance_foot_acceleration']=max(stance_acc)<tol
    if config.get('max_crouch_com_drop_m') is not None:
        metrics['stance_com_drop_m']=float(initial_com[2]-coms[:takeoff+1,2].min())
        checks['crouch_depth']=metrics['stance_com_drop_m']<=config['max_crouch_com_drop_m']+tol
    if config.get('min_knee_center_height_m') is not None:
        metrics['min_stance_knee_center_height_m']=float(min(stance_knees))
        checks['knee_center_clearance']=min(stance_knees)>=config['min_knee_center_height_m']-tol
    if config.get('max_base_pitch_deg') is not None or config.get('max_world_omega_y_rad_s') is not None:
        from scipy.spatial.transform import Rotation
        rotations=Rotation.from_quat(np.array([n['q'][3:7] for n in nodes]))
        metrics['max_abs_base_pitch_deg']=float(np.max(abs(rotations.as_euler('xyz',degrees=True)[:,1])))
        metrics['max_abs_world_omega_y_rad_s']=float(np.max(abs(rotations.apply(np.array([n['v'][3:6] for n in nodes]))[:,1])))
        if config.get('max_base_pitch_deg') is not None:
            checks['base_pitch_limit']=metrics['max_abs_base_pitch_deg']<=config['max_base_pitch_deg']+np.rad2deg(tol)
        if config.get('max_world_omega_y_rad_s') is not None:
            checks['world_omega_y_limit']=metrics['max_abs_world_omega_y_rad_s']<=config['max_world_omega_y_rad_s']+tol
    if config.get('max_landing_com_drop_from_final_m') is not None:
        metrics['landing_com_drop_from_final_m']=float(coms[-1,2]-coms[land:,2].min())
        checks['landing_buffer_depth']=metrics['landing_com_drop_from_final_m']<=config['max_landing_com_drop_from_final_m']+tol
    if config.get('max_flight_pitch_deg') is not None:
        from scipy.spatial.transform import Rotation
        angles=Rotation.from_quat(np.array([n['q'][3:7] for n in nodes[takeoff:land+1]])).as_euler('xyz',degrees=True)[:,1]
        metrics['max_abs_flight_pitch_deg']=float(np.max(abs(angles)))
        checks['flight_pitch_limit']=metrics['max_abs_flight_pitch_deg']<=config['max_flight_pitch_deg']+np.rad2deg(tol)
    if config.get('main_joint_planning_limit_Nm') is not None:
        metrics['max_main_joint_torque_Nm']=max(j['peak_abs_Nm'] for j in report['joint_torques'] if any(part in j['joint'] for part in ('hip','thigh','shank')))
        metrics['main_joint_hardware_headroom_Nm']=min(j['limit_Nm']-j['peak_abs_Nm'] for j in report['joint_torques'] if any(part in j['joint'] for part in ('hip','thigh','shank')))
        checks['main_joint_planning_torque']=metrics['max_main_joint_torque_Nm']<=config['main_joint_planning_limit_Nm']+tol
    if config.get('flight_pitch_range_deg') is not None:
        from scipy.spatial.transform import Rotation
        angles=Rotation.from_quat(np.array([n['q'][3:7] for n in nodes[takeoff:land+1]])).as_euler('xyz',degrees=True)[:,1]
        metrics['flight_pitch_range_deg']=[float(angles.min()),float(angles.max())]
        lower,upper=config['flight_pitch_range_deg']
        checks['flight_pitch_band']=angles.min()>=lower-np.rad2deg(tol) and angles.max()<=upper+np.rad2deg(tol)
    checks={name:bool(value) for name,value in checks.items()}
    if config.get('landing_stopping_margin_m') is not None:
        decel=2*float(np.linalg.norm(model.gravity.linear))
        if abs(config['landing_vertical_deceleration_bound_m_s2']-decel)>1e-10:
            raise ValueError('Independent stopping audit expects the current 1.5 mg per-wheel force bound')
        margin=coms[land:,2]-(coms[-1,2]-config['max_landing_com_drop_from_final_m'])-np.minimum(vs[land:,2],0.)**2/(2*decel)
        metrics['min_landing_stopping_margin_m']=float(min(margin))
        checks['landing_stopping_distance_necessary_bound']=bool(min(margin)>=config['landing_stopping_margin_m']-tol)
    if config.get('wheel_x_corridor_half_width_m') is not None:
        distance=max(abs(x-terminal_mid[0]) for x in support_wheel_x)
        metrics['max_wheel_x_distance_from_terminal_mid_m']=float(distance)
        checks['whole_support_wheel_x_corridor']=bool(distance<=config['wheel_x_corridor_half_width_m']+tol)
    from audit_foot_recovery import recovery_metrics
    csv_time_file=csv_path.parent/'csv_optimizer_times_s.csv'
    csv_times=np.loadtxt(csv_time_file,skiprows=1) if csv_time_file.exists() else None
    recovery,_,_=recovery_metrics(saved,model,csv_q,csv_times)
    metrics['foot_recovery']=recovery
    checks['node_joint_position_bounds']=recovery['max_node_joint_position_violation_rad']<=tol
    checks['csv_joint_position_bounds']=recovery['max_csv_joint_position_violation_rad']<=tol
    checks['foot_hermite_position_bounds']=max(f['hermite_limit_violation_rad'] for f in recovery['feet'].values())<=tol
    checks['foot_hermite_velocity_bounds']=all(f['hermite_max_abs_velocity_rad_s']<=f['velocity_limit_rad_s']+tol for f in recovery['feet'].values())
    if config.get('foot_position_margin_deg',0):
        from audit_foot_recovery import hermite_extrema
        margins=[];csv_margins=[]
        for side in ('left','right'):
            j=model.joints[model.getJointId(side+'_foot_joint')]
            csv_margins.extend([float(np.min(csv_q[:,j.idx_q])-model.lowerPositionLimit[j.idx_q]),
                                float(model.upperPositionLimit[j.idx_q]-np.max(csv_q[:,j.idx_q]))])
            for n,nxt in zip(nodes[:-1],nodes[1:]):
                lo,hi=hermite_extrema(n['q'][j.idx_q],nxt['q'][j.idx_q],n['v'][j.idx_v],nxt['v'][j.idx_v],n['dt'])
                margins.extend([lo-model.lowerPositionLimit[j.idx_q],model.upperPositionLimit[j.idx_q]-hi])
        metrics['foot_minimum_position_margin_deg']=float(np.rad2deg(min(margins)))
        checks['foot_continuous_position_margin']=bool(min(margins)>=np.deg2rad(config['foot_position_margin_deg'])-tol)
        metrics['csv_foot_minimum_position_margin_deg']=float(np.rad2deg(min(csv_margins)))
        checks['csv_foot_position_margin']=bool(min(csv_margins)>=np.deg2rad(config['foot_position_margin_deg'])-tol)
    if 'terminal_target_q' in config:
        error=np.array(nodes[-1]['q'])-np.array(config['terminal_target_q'])
        if config.get('free_terminal_wheel_angles'):
            for side in ('left','right'):error[model.joints[model.getJointId(side+'_wheel_joint')].idx_q]=0.
        metrics['terminal_target_max_coordinate_error']=float(np.max(abs(error)))
        checks['configured_terminal_pose']=bool(np.max(abs(error))<=tol)
    if config.get('post_touchdown_backward_limit_m') is not None:
        checks['wheel_backward_excursion']=max(s['backward_excursion_m'] for s in recovery['wheel_node_travel'].values())<=config['post_touchdown_backward_limit_m']+tol
        checks['csv_wheel_backward_excursion']=bool(csv_times is not None and max(recovery['csv_wheel_backward_excursion_m'].values())<=config['post_touchdown_backward_limit_m']+tol)
    if config.get('touchdown_joint_ranges_deg'):
        checks['touchdown_foot_angles']=all(lo-np.rad2deg(tol)<=recovery['feet'][name]['touchdown_deg']<=hi+np.rad2deg(tol)
                                           for name,(lo,hi) in config['touchdown_joint_ranges_deg'].items())
    from audit_wheel_motion import wheel_motion_metrics
    wheel_motion=wheel_motion_metrics(saved,model,csv_q,csv_times)
    metrics['wheel_motion']=wheel_motion
    checks['wheel_node_speed_bounds']=all(m['node_peak_speed_rad_s']<=m['speed_limit_rad_s']+tol for m in wheel_motion.values())
    checks['wheel_hermite_speed_bounds']=all(m['hermite_peak_speed_rad_s']<=m['speed_limit_rad_s']+tol for m in wheel_motion.values())
    if config.get('airborne_wheel_speed_limit_rad_s') is not None:
        limit=config['airborne_wheel_speed_limit_rad_s']
        checks['airborne_wheel_node_speed_bounds']=all(m['airborne_node_peak_speed_rad_s']<=limit+tol for m in wheel_motion.values())
        checks['airborne_wheel_hermite_speed_bounds']=all(m['airborne_hermite_peak_speed_rad_s']<=limit+tol for m in wheel_motion.values())
    if config.get('post_touchdown_excursion_limit_m') is not None:
        limit=config['post_touchdown_excursion_limit_m']
        checks['wheel_touchdown_relative_excursion']=all(m['max_abs_touchdown_relative_x_m']<=limit+tol for m in wheel_motion.values())
        checks['csv_wheel_touchdown_relative_excursion']=all(m.get('csv_max_abs_touchdown_relative_x_m',np.inf)<=limit+tol for m in wheel_motion.values())
    result={'metrics':metrics,'checks':checks,'all_independent_checks_pass':all(checks.values()),'tolerance':tol,
            'csv_training_ready':False,'note':'Physical point/flight checks do not establish interpolation or closed-loop contact feasibility.',
            'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (solution,csv_path,URDF,Path(__file__))}}
    output.mkdir(parents=True,exist_ok=False);write_json(output/'validation.json',result)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('solution',type=Path);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.argv=[sys.argv[0]];validate(args.solution,args.csv,args.output)
