#%% import
import numpy as np
from manimlib.imports import *

# camera angle:
# phi - yz plane
# theta - xy plane
# gamma - xz plane
#%% monkey patching
def phi_of_vector(vector):
    xy = complex(*vector[:2])
    if xy == 0:
        return 0
    a = ((vector[:1]) ** 2 + (vector[1:2]) ** 2) ** (1 / 2)
    vector[0] = a
    vector[1] = vector[2]
    return np.angle(complex(*vector[:2]))


def position_tip(self, tip, at_start=False):
    if at_start:
        anchor = self.get_start()
        handle = self.get_first_handle()
    else:
        handle = self.get_last_handle()
        anchor = self.get_end()
    tip.rotate(angle_of_vector(handle - anchor) - PI - tip.get_angle())
    angle = angle_of_vector(handle - anchor) + PI / 2
    a = np.array([np.cos(angle), np.sin(angle), 0])
    tip.rotate(-phi_of_vector(handle - anchor), a)
    tip.shift(anchor - tip.get_tip_point())
    return tip


TipableVMobject.position_tip = position_tip

#%% common settings and setup
kw_cam = {
    "phi": 1 / 3 * PI,
    "theta": 1 / 5 * PI,
    "distance": 2.5,
    "frame_center": np.array([0, 0, 0]),
}
kw_plane = {"x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2}
kw_axes = {
    "x_min": -2,
    "x_max": 2,
    "y_min": -2,
    "y_max": 2,
    "z_min": -2,
    "z_max": 2,
    "axis_config": {"include_tip": True},
}
kw_vec_init = {"color": COLOR_MAP["GOLD_D"]}
kw_vec = {"color": COLOR_MAP["BLUE_D"]}
kw_lab = {"should_center": True}
kw_line = {"stroke_width": 2}
color_dict = {
    1: COLOR_MAP["RED_E"],
    2: COLOR_MAP["RED_D"],
    3: COLOR_MAP["RED_C"],
    4: COLOR_MAP["RED_B"],
    5: COLOR_MAP["RED_A"],
    6: COLOR_MAP["RED_A"],
}
ball_dict = {
    (1, 1, -1): COLOR_MAP["MAROON_E"],
    (1, -1, 1): COLOR_MAP["TEAL_E"],
    (-1, 1, 1): COLOR_MAP["PURPLE_C"],
}


def setup_crd():
    cam = ThreeDCamera(**kw_cam)
    axes = ThreeDAxes(**kw_axes)
    xlab = axes.get_axis_label(
        r"\text{red}", axes.get_x_axis(), edge=RIGHT, direction=ORIGIN
    )
    xlab.shift((0.15, 0, 0))
    ylab = axes.get_axis_label(
        r"\text{round}", axes.get_y_axis(), edge=UP, direction=ORIGIN
    )
    ylab.shift((0, 0.15, 0))
    zlab = axes.get_axis_label(
        r"\text{soft}", axes.get_z_axis(), edge=OUT, direction=ORIGIN
    )
    zlab.shift((0, 0, 0.15))
    axes.add(xlab, ylab, zlab)
    plane = NumberPlane(**kw_plane)
    return cam, axes, plane, (xlab, ylab, zlab)


eig = np.array([[1, 1, -1], [1, -1, 1], [-1, 1, 1]])
eig_inv = np.linalg.inv(eig)
W = np.dot(np.dot(eig, np.diag(np.array([3, 1, 1.5]))), eig_inv)


class ScnCommon(ThreeDScene):
    def move_vec(
        self,
        vec,
        lab,
        loc,
        sh,
        t,
        lab_new=None,
        kw_vec=dict(),
        kw_lab=dict(),
        kw_line=dict(),
    ):
        vec_new = Vector(loc, **kw_vec)
        trace = Line(vec.get_end(), loc, **kw_line)
        if lab_new is not None:
            lab_new = self.lab_vec(vec_new, lab_new, sh, **kw_lab)
            self.play(
                ReplacementTransform(vec, vec_new),
                FadeOut(lab),
                ShowCreation(lab_new),
                ShowCreation(trace),
                run_time=t,
            )
        else:
            lab_loc = loc + sh * np.sign(vec.get_end())
            self.play(
                ReplacementTransform(vec, vec_new),
                ApplyMethod(lab.move_to, lab_loc),
                ShowCreation(trace),
                run_time=t,
            )
        return vec_new, lab_new, trace

    def set_coords(self, obj, crd):
        for dim, c in enumerate(crd):
            obj.set_coord(c, dim)

    def lab_vec(self, vec, lab, sh, **kw_lab):
        tip = vec.get_end()
        sh = sh * np.sign(tip)
        lab = TexMobject(lab, **kw_lab)
        self.set_coords(lab, tip + sh)
        self.add_fixed_orientation_mobjects(lab)
        return lab


#%% raw scene
class ScnRaw(ScnCommon):
    def construct(self):
        # setup
        cam, axes, plane, labs = setup_crd()
        self.set_camera(cam)
        self.add(axes)
        self.add(plane)
        self.add_fixed_orientation_mobjects(*labs)
        # init
        vec_apple = Vector((1, 1, -1), **kw_vec_init)
        vec_elmo = Vector((1, -1, 1), **kw_vec_init)
        lab_apple = self.lab_vec(vec_apple, r"\vec{apple}", 0.2)
        lab_elmo = self.lab_vec(vec_elmo, r"\vec{elmo}", 0.2)
        group_init = VGroup(vec_apple, vec_elmo, lab_apple, lab_elmo)
        self.play(ShowCreation(group_init))
        self.wait(1)
        # linear
        for nvec in range(10):
            x = np.random.random(size=3) * 2 - 1
            vec_x = Vector(x, **kw_vec)
            lab_x = self.lab_vec(vec_x, r"\vec{x}_0", 0.2)
            vec_group = VGroup(vec_x, lab_x)
            self.play(ShowCreation(vec_group))
            trace_ls = []
            for it in range(1, 5):
                x = np.dot(W, x)
                lab_new = r"\mathbf{W}" + lab_x.get_tex_string()
                kw_line["stroke_color"] = color_dict[it]
                x_len = np.linalg.norm(x) ** 2
                if x_len > 2.5:
                    self.move_camera(distance=x_len, run_time=0.3)
                vec_x, lab_x, trace = self.move_vec(
                    vec=vec_x,
                    lab=lab_x,
                    loc=x,
                    sh=0.2,
                    t=0.5,
                    lab_new=lab_new,
                    kw_vec=kw_vec,
                    kw_line=kw_line,
                )
                vec_group = VGroup(vec_x, lab_x)
                trace_ls.append(trace)
                self.wait(0.5)
            self.play(FadeOut(vec_group), FadeOut(VGroup(*trace_ls)))
            self.move_camera(distance=2.5)


#%% linear scene
class ScnLinear(ScnCommon):
    def construct(self):
        # setup
        cam, axes, plane, labs = setup_crd()
        self.set_camera(cam)
        self.add(axes)
        self.add(plane)
        self.add_fixed_orientation_mobjects(*labs)
        # init
        vec_apple = Vector((1, 1, -1), **kw_vec_init)
        vec_elmo = Vector((1, -1, 1), **kw_vec_init)
        lab_apple = self.lab_vec(vec_apple, r"\vec{apple}", 0.2)
        lab_elmo = self.lab_vec(vec_elmo, r"\vec{elmo}", 0.2)
        group_init = VGroup(vec_apple, vec_elmo, lab_apple, lab_elmo)
        self.play(ShowCreation(group_init), run_time=0.3)
        self.wait(1)
        # linear
        self.begin_ambient_camera_rotation(rate=0.02)
        for nvec in range(22):
            x = np.random.random(size=3) * 2 - 1
            x_norm = x / np.linalg.norm(x)
            vec_x = Vector(x_norm, **kw_vec)
            lab_x = self.lab_vec(vec_x, r"\vec{x}_0", 0.2)
            vec_group = VGroup(vec_x, lab_x)
            self.play(ShowCreation(vec_group), run_time=0.3)
            for it in range(1, 6):
                x = np.dot(W, x)
                lab_new = r"\vec{x}_" + str(it)
                x_norm = x / np.linalg.norm(x) * np.sqrt(3)
                kw_line["stroke_color"] = color_dict[it]
                vec_x, lab_x, trace = self.move_vec(
                    vec=vec_x,
                    lab=lab_x,
                    loc=x_norm,
                    sh=0.2,
                    t=0.3,
                    lab_new=lab_new,
                    kw_vec=kw_vec,
                    kw_line=kw_line,
                )
                vec_group = VGroup(vec_x, lab_x)
                self.wait(0.3)
            self.play(FadeOut(vec_group), run_time=0.1)
        self.begin_ambient_camera_rotation(rate=-0.2 * PI)
        self.wait(5)


#%% sign scene
class ScnSign(ScnCommon):
    def construct(self):
        # setup
        cam, axes, plane, labs = setup_crd()
        cam = ThreeDCamera(phi=1 / 3 * PI, theta=0, distance=2.5,)
        self.set_camera(cam)
        self.add(axes)
        self.add(plane)
        self.add_fixed_orientation_mobjects(*labs)
        # init
        vec_apple = Vector((1, 1, -1), **kw_vec_init)
        vec_elmo = Vector((1, -1, 1), **kw_vec_init)
        lab_apple = self.lab_vec(vec_apple, r"\vec{apple}", 0.2)
        lab_elmo = self.lab_vec(vec_elmo, r"\vec{elmo}", 0.2)
        group_init = VGroup(vec_apple, vec_elmo, lab_apple, lab_elmo)
        self.play(ShowCreation(group_init), run_time=0.3)
        self.wait(1)
        # linear
        self.begin_ambient_camera_rotation(rate=0.02)
        for nvec in range(50):
            x = np.random.random(size=3) * 2 - 1
            vec_x = Vector(x, **kw_vec)
            lab_x = self.lab_vec(vec_x, r"\vec{x}_0", 0.2)
            vec_group = VGroup(vec_x, lab_x)
            self.play(ShowCreation(vec_group), run_time=0.3)
            for it in range(1, 3):
                xnew = np.sign(np.dot(W, x))
                if (xnew == x).all():
                    break
                x = xnew
                lab_new = r"\vec{x}_" + str(it)
                kw_line["stroke_color"] = color_dict[it]
                vec_x, lab_x, trace = self.move_vec(
                    vec=vec_x,
                    lab=lab_x,
                    loc=x,
                    sh=0.2,
                    t=0.3,
                    lab_new=lab_new,
                    kw_vec=kw_vec,
                    kw_line=kw_line,
                )
                vec_group = VGroup(vec_x, lab_x)
                self.wait(0.3)
            self.play(FadeOut(vec_group), run_time=0.1)
        self.begin_ambient_camera_rotation(rate=-0.2 * PI)
        self.wait(5)


#%% stochastic
class ScnStochastic(ScnCommon):
    def construct(self):
        # setup
        cam, axes, plane, labs = setup_crd()
        cam = ThreeDCamera(phi=1 / 3 * PI, theta=0, distance=2.5,)
        self.set_camera(cam)
        self.add(axes)
        self.add(plane)
        self.add_fixed_orientation_mobjects(*labs)
        # init
        vec_apple = Vector((1, 1, -1), **kw_vec_init)
        vec_elmo = Vector((1, -1, 1), **kw_vec_init)
        lab_apple = self.lab_vec(vec_apple, r"\vec{apple}", 0.2)
        lab_elmo = self.lab_vec(vec_elmo, r"\vec{elmo}", 0.2)
        group_init = VGroup(vec_apple, vec_elmo, lab_apple, lab_elmo)
        self.play(ShowCreation(group_init), run_time=0.3)
        self.wait(1)
        # linear
        self.begin_ambient_camera_rotation(rate=0.01)
        for nvec in range(51):
            x = np.random.random(size=3) * 2 - 1
            vec_x = Vector(x, **kw_vec)
            lab_x = self.lab_vec(vec_x, r"\vec{x}_0", 0.2)
            vec_group = VGroup(vec_x, lab_x)
            self.play(ShowCreation(vec_group), run_time=0.3)
            for it in range(1, 6):
                wx = np.sign(np.dot(W, x))
                if (wx == x).all():
                    break
                idx = np.random.randint(3)
                x[idx] = wx[idx]
                lab_new = r"\vec{x}_" + str(it)
                kw_line["stroke_color"] = color_dict[it]
                vec_x, lab_x, trace = self.move_vec(
                    vec=vec_x,
                    lab=lab_x,
                    loc=x,
                    sh=0.2,
                    t=0.1,
                    lab_new=lab_new,
                    kw_vec=kw_vec,
                    kw_line=kw_line,
                )
                vec_group = VGroup(vec_x, lab_x)
                self.wait(0.2)
            self.play(FadeOut(vec_group), run_time=0.3)
        self.begin_ambient_camera_rotation(rate=-0.2 * PI)
        self.wait(5)


#%% ball
class ScnBall(ScnCommon):
    def construct(self):
        # setup
        kw_line["stroke_opacity"] = 0.6
        cam, axes, plane, labs = setup_crd()
        cam = ThreeDCamera(phi=1 / 3 * PI, theta=0, distance=4.3,)
        self.set_camera(cam)
        self.add(axes)
        self.add(plane)
        self.add_fixed_orientation_mobjects(*labs)
        # init
        kw_vec_init["stroke_width"] = 3
        kw_vec_init["color"] = ball_dict[(1, 1, -1)]
        vec_apple = Vector((1, 1, -1), **kw_vec_init)
        vec_apple_ = Vector((-1, -1, 1), **kw_vec_init)
        kw_vec_init["color"] = ball_dict[(1, -1, 1)]
        vec_elmo = Vector((1, -1, 1), **kw_vec_init)
        vec_elmo_ = Vector((-1, 1, -1), **kw_vec_init)
        kw_vec_init["color"] = ball_dict[(-1, 1, 1)]
        vec_yolk = Vector((-1, 1, 1), **kw_vec_init)
        vec_yolk_ = Vector((1, -1, -1), **kw_vec_init)
        lab_apple = self.lab_vec(vec_apple, r"\vec{apple}", 0.2)
        lab_elmo = self.lab_vec(vec_elmo, r"\vec{elmo}", 0.2)
        lab_yolk = self.lab_vec(vec_yolk, r"\vec{yolk}", 0.2)
        group_init = VGroup(
            vec_apple,
            vec_elmo,
            vec_yolk,
            lab_apple,
            lab_elmo,
            lab_yolk,
            vec_apple_,
            vec_elmo_,
            vec_yolk_,
        )
        self.play(ShowCreation(group_init), run_time=1)
        self.wait(2)
        # linear
        self.begin_ambient_camera_rotation(rate=0.02)
        for nvec in range(81):
            X = np.random.random(size=(3, 10)) * 4 - 2
            x_ls = [Vector(x, **kw_vec) for x in X.T]
            vec_X = VGroup(*x_ls)
            self.play(ShowCreation(vec_X), run_time=0.3)
            Xnew = np.sign(np.dot(W, np.sign(np.dot(W, X))))
            xnew_ls = [Vector(x, **kw_vec) for x in Xnew.T]
            vec_xnew = VGroup(*xnew_ls)
            trans_ls = []
            trace_ls = []
            for x, xnew in zip(x_ls, xnew_ls):
                trans_ls.append(ReplacementTransform(x, xnew))
                try:
                    kw_line["color"] = ball_dict[tuple(xnew.get_end())]
                except KeyError:
                    kw_line["color"] = ball_dict[tuple(-xnew.get_end())]
                trace = Line(x.get_end(), xnew.get_end(), **kw_line)
                trace_ls.append(ShowCreation(trace))
            self.play(*trans_ls, *trace_ls, run_time=0.3)
            self.play(FadeOut(vec_xnew), run_time=0.1)
        self.begin_ambient_camera_rotation(rate=-0.2 * PI)
        self.wait(10)
