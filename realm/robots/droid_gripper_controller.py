import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControlType, GripperController, IsGraspingState
from omnigibson.macros import create_module_macros
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

VALID_MODES = {
    "binary",
    "smooth",
    "independent",
}


# Create settings for this module
#m = create_module_macros(module_path=__file__)

# is_grasping heuristics parameters
POS_TOLERANCE = 0.002  # arbitrary heuristic
VEL_TOLERANCE = 0.01  # arbitrary heuristic


class MultiFingerGripperController(GripperController):
    """
    Controller class for multi finger gripper control. This either interprets an input as a binary
    command (open / close), continuous command (open / close with scaled velocities), or per-joint continuous command

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. Convert command into gripper joint control signals
        2b. Clips the resulting control by the motor limits

    NOTE on porting to newer OmniGibson (see PLAN.md / thesis Methodology for the full rationale):
    the previous single-instance `control_dict` parameter has been removed from the controller
    framework in favor of a batched, multi-robot-capable design (`ControllableObjectViewAPI`,
    `self.routing_path`, `self.view_row_indices`). This class only overrides the methods REALM's
    original DROID gripper controller overrode; every method it does NOT override (`add_member`,
    `_dump_state`/`_load_state`/`serialize`/`deserialize`, ...) is inherited unmodified from
    `GripperController`/`ControllerBase` and works correctly as-is -- no porting needed there.
    Since our evaluation setup always has exactly one robot (N=1), state that the native class
    stores per-member as a list (e.g. `_is_grasping`) is kept as a plain scalar here for
    simplicity; `compute_control` still operates on the batched (N, dim) array shape the
    framework passes in, since that shape is fixed by the framework regardless of N.
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        inverted=False,
        mode="binary",
        open_qpos=None,
        closed_qpos=None,
        limit_tolerance=0.001,
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            motor_type (str): type of motor being controlled, one of {position, velocity, effort}
            control_limits (Dict[str, Tuple[Array[float], Array[float]]]): The min/max limits to the outputted
                control signal. Should specify per-dof type limits, i.e.:

                "position": [[min], [max]]
                "velocity": [[min], [max]]
                "effort": [[min], [max]]
                "has_limit": [...bool...]

                Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
            dof_idx (Array[int]): specific dof indices controlled by this robot. Used for inferring
                controller-relevant values during control computations
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max acceptable inputted command. Values outside this range will be clipped.
                If None, no clipping will be used. If "default", range will be set to (-1, 1)
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max scaled command. If both this value and @command_input_limits is not None,
                then all inputted command values will be scaled from the input range to the output range.
                If either is None, no scaling will be used. If "default", then this range will automatically be set
                to the @control_limits entry corresponding to self.control_type
            inverted (bool): whether or not the command direction (grasp is negative) and the control direction are
                inverted, e.g. to grasp you need to move the joint in the positive direction.
            mode (str): mode for this controller. Valid options are:

                "binary": 1D command, if preprocessed value > 0 is interpreted as an max open
                    (send max pos / vel / tor signal), otherwise send max close control signals
                "smooth": 1D command, sends symmetric signal to all finger joints equal to the preprocessed commands
                "independent": n-dimensional command, sends independent signals to each finger joint equal to the preprocessed command

            open_qpos (None or Array[float]): If specified, the joint positions representing a fully-opened gripper.
                This is to allow representing the open state as a partially opened gripper, rather than the full
                opened gripper. If None, will simply use the native joint limits of the gripper joints. Only relevant
                if using @mode=binary and @motor_type=position
            closed_qpos (None or Array[float]): If specified, the joint positions representing a fully-closed gripper.
                This is to allow representing the closed state as a partially closed gripper, rather than the full
                closed gripper. If None, will simply use the native joint limits of the gripper joints. Only relevant
                if using @mode=binary and @motor_type=position
            limit_tolerance (float): sets the tolerance from the joint limit ends, below which controls will be zeroed
                out if the control is using velocity or torque control
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self._motor_type = motor_type.lower()
        assert_valid_key(key=mode, valid_keys=VALID_MODES, name="mode for multi finger gripper")
        self._inverted = inverted
        self._mode = mode
        self._limit_tolerance = limit_tolerance
        self._open_qpos = open_qpos if open_qpos is None else th.tensor(open_qpos)
        self._closed_qpos = closed_qpos if closed_qpos is None else th.tensor(closed_qpos)

        # Create other args to be filled in at runtime
        self._is_grasping = IsGraspingState.FALSE
        # NOTE: patched -- tracks the last control signal applied (single-instance N=1
        # shortcut, see class docstring), used by _update_grasping_state to detect "no control
        # issued yet". Was referenced but never actually initialized/updated in an earlier
        # revision of this port (AttributeError: no attribute '_control').
        self._control = None

        # If we're using binary signal, we override the command output limits
        if mode == "binary":
            command_output_limits = (-1.0, 1.0)

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self, controller_idx):
        # NOTE: patched -- newer OmniGibson's ControllerView batches multiple controller
        # instances per group and always calls reset(controller_idx); the old single-instance
        # signature (no args) no longer matches the base class.
        super().reset(controller_idx)
        self._is_grasping = IsGraspingState.FALSE
        self._control = None

    def _preprocess_command(self, command):
        # We extend this method to make sure command is always n-dimensional
        if self._mode != "independent":
            command = (
                th.tensor([command] * self.command_dim)
                if type(command) in {int, float}
                else th.tensor([command[0]] * self.command_dim)
            )

        # Flip the command if the direction is inverted.
        if self._inverted:
            command = self._command_input_limits[1] - (command - self._command_input_limits[0])

        # Return from super method
        return super()._preprocess_command(command=command)

    def _update_goal(self, controller_idx, command):
        # NOTE: patched signature (controller_idx, command) instead of (command, control_dict);
        # control_dict was never used here, so no other change needed.
        return dict(target=command)

    def compute_control(self, goals):
        """
        Converts the (already preprocessed) batched goal into a deployable (non-clipped!) gripper
        joint control signal.

        NOTE: patched -- `control_dict` is gone; current joint state now comes from
        `ControllableObjectViewAPI` via `self.routing_path`/`self.view_row_indices`. The framework
        always calls this with a batched (N, dim) array regardless of N, so even for our
        single-robot (N=1) case the code below operates on shape (1, dim) arrays throughout.

        Args:
            goals (Dict[str, Any]): dictionary of batched goals. Must include:
                    target: (N, command_dim) desired gripper target

        Returns:
            Array[float]: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        target = goals["target"]  # (N, command_dim)

        rows = self.view_row_indices
        # NOTE: ControllableObjectViewAPI may return numpy arrays depending on the active compute
        # backend; wrap with th.as_tensor() before any torch-only ops.
        joint_pos = th.as_tensor(ControllableObjectViewAPI.get_all_joint_positions(self.routing_path))[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        # Choose what to do based on control mode
        if self._mode == "binary":
            should_open = target[:, 0] >= 0.0 if not self._inverted else target[:, 0] > 0.0  # (N,)
            open_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx]
                if self._open_qpos is None
                else self._open_qpos
            )  # (ctrl_dim,)
            closed_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx]
                if self._closed_qpos is None
                else self._closed_qpos
            )  # (ctrl_dim,)
            u = th.where(should_open[:, None], open_limit, closed_limit)  # (N, ctrl_dim)
            # NOTE: removed here -- REALM's original code additionally did
            # `u[2:] = joint_pos[:2] / 0.05 * 0.785` to manually drive the *outer* finger
            # joints of its custom 4-DOF DROID/Robotiq-style gripper. The native "franka"
            # gripper we now use only has 2 DOF (standard parallel-jaw fingers, no separate
            # outer-finger joints), so that line no longer applies and would index out of
            # bounds; dropped as part of the DROID->franka substitution (see thesis Methodology
            # caveat on using the stock Franka model).
        else:
            # Use continuous signal. Make sure to go from command to control dim.
            u = target * th.ones(self.control_dim) if target.shape[1] == 1 else target

        # If we're near the joint limits and we're using velocity / torque control, we zero out the action
        if self._motor_type in {"velocity", "torque"}:
            pos_hi = self._control_limits[ControlType.POSITION][1][self.dof_idx]  # (ctrl_dim,)
            pos_lo = self._control_limits[ControlType.POSITION][0][self.dof_idx]  # (ctrl_dim,)
            violate_upper_limit = joint_pos > pos_hi - self._limit_tolerance  # (N, ctrl_dim)
            violate_lower_limit = joint_pos < pos_lo + self._limit_tolerance  # (N, ctrl_dim)
            violation = (violate_upper_limit & (u > 0)) | (violate_lower_limit & (u < 0))
            u = u * ~violation

        # Update whether we're grasping or not (single-instance shortcut: N==1, take row 0)
        self._update_grasping_state(joint_pos[0], u[0])

        # Return control
        return u

    def _update_grasping_state(self, joint_pos, control):
        """
        Updates internal inferred grasping state of the gripper being controlled by this gripper controller.

        NOTE: patched signature -- takes the already-fetched current joint positions and applied
        control directly (single-instance, N=1 shortcut) instead of a `control_dict`.

        Args:
            joint_pos (Array): current joint positions for this controller's dof_idx
            control (Array): the control signal that was just computed/applied
        """
        # Independent mode of MultiFingerGripperController does not have any good heuristics to determine is_grasping
        if self._mode == "independent":
            is_grasping = IsGraspingState.UNKNOWN

        # No control has been issued before -- we assume not grasping
        elif self._control is None:
            is_grasping = IsGraspingState.FALSE

        #  Different values in the command for non-independent mode - cannot use heuristics
        elif not th.all(control == control[0]):
            is_grasping = IsGraspingState.UNKNOWN

        # Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the gripper
        # controller's tolerance of zero-ing out velocities, which makes the heuristics invalid.
        elif not POS_TOLERANCE > self._limit_tolerance:
            is_grasping = IsGraspingState.UNKNOWN

        else:
            finger_pos = joint_pos

            # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
            if self._motor_type == "position" and th.mean(th.abs(finger_pos - control)) < POS_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # For joint velocity / torque control, if the desired velocities / torques are zeros, is_grasping unknown
            elif self._motor_type in {"velocity", "torque"} and th.mean(th.abs(control)) < VEL_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # Otherwise, the last control signal intends to "move" the gripper
            else:
                rows = self.view_row_indices
                finger_vel = th.as_tensor(
                    ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)
                )[rows, :][:, self.dof_idx][0]
                min_pos = self._control_limits[ControlType.POSITION][0][self.dof_idx]
                max_pos = self._control_limits[ControlType.POSITION][1][self.dof_idx]

                # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                finger_pos = th.clip(finger_pos, min_pos, max_pos)

                # Check distance from both ends of the joint limits
                dist_from_lower_limit = finger_pos - min_pos
                dist_from_upper_limit = max_pos - finger_pos

                # If the joint positions are not near the joint limits with some tolerance (POS_TOLERANCE)
                valid_grasp_pos = (
                    th.mean(dist_from_lower_limit) > POS_TOLERANCE
                    and th.mean(dist_from_upper_limit) > POS_TOLERANCE
                )

                # And the joint velocities are close to zero with some tolerance (VEL_TOLERANCE)
                valid_grasp_vel = th.all(th.abs(finger_vel) < VEL_TOLERANCE)

                # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                is_grasping = IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE

        # Store calculated state
        self._is_grasping = is_grasping
        self._control = control

    def compute_no_op_goal(self, controller_idx):
        # NOTE: patched -- was compute_no_op_goal(self, control_dict); current joint position now
        # comes from ControllableObjectViewAPI, keyed by this member's prim_path.
        if self._mode == "binary":
            return dict(target=th.zeros(self.command_dim))

        prim_path = self._articulation_root_paths[controller_idx]
        if self._motor_type == "position":
            target = th.as_tensor(ControllableObjectViewAPI.get_joint_positions(prim_path))[self.dof_idx]
        elif self._motor_type == "velocity":
            target = th.zeros(self.command_dim)
        else:
            raise ValueError("Cannot compute noop action for effort motor type.")

        if self._mode == "smooth":
            target = th.mean(target, dim=-1, keepdim=True)

        return dict(target=target)

    def _compute_no_op_action(self, controller_idx):
        # NOTE: patched -- was _compute_no_op_action(self, control_dict).
        if self._mode == "binary":
            command_val = -1 if self.is_grasping() == IsGraspingState.TRUE else 1
            if self._inverted:
                command_val = -1 * command_val
            return th.tensor([command_val], dtype=th.float32)

        prim_path = self._articulation_root_paths[controller_idx]
        if self._motor_type == "position":
            command = th.as_tensor(ControllableObjectViewAPI.get_joint_positions(prim_path))[self.dof_idx]
        elif self._motor_type == "velocity":
            command = th.zeros(self.command_dim)
        else:
            raise ValueError("Cannot compute noop action for effort motor type.")

        # Convert to binary / smooth mode if necessary
        if self._mode == "smooth":
            command = th.mean(command, dim=-1, keepdim=True)

        return command

    def _get_goal_shapes(self):
        return dict(target=(self.command_dim,))

    def is_grasping(self, controller_idx=0):
        # NOTE: patched signature -- added controller_idx=0 (native base class now always calls
        # with an index; N=1 shortcut means we ignore its value and return the single cached state).
        return self._is_grasping

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx) if self._mode == "independent" else 1
