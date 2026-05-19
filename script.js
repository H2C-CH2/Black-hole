// === Physical Constants ===
const c = 299792458.0; // meters/second
const G = 6.6743e-11; // m^3 kg^-1 s^-2
const e = 8.854e-12; // Farads/meter
const k = 8.98755e9; // N m^2/C^2

// === Variables

const M_SUN = 1.98892e30; // kg

let M = 4.3e6 * M_SUN;   // solar masses
let aStar = 0.0; // dimensionless spin  a* = a/M  in [0,1)
let QStar = 0.0; // dimensionless charge in [0,1)

let Rs, scale;

// === Simulation State ===
let objects = [
    { pos: [4e11, 0, 0], radius: 4e10, color: [1, 1, 0, 1], mass: 1.98892e30 },
    { pos: [0, 0, 4e11], radius: 4e10, color: [1, 0.3, 0, 1], mass: 1.98892e30 },
    { pos: [0, 0, 0], radius: Rs, color: [0, 0, 0, 1], mass: M },
];

function updateBHParams(massSolar, aStarIn, qStarIn) {
    M = massSolar * M_SUN;
    aStar = aStarIn;
    QStar = qStarIn;
    Rs = (2 * G * M) / (c * c);
    scale = 1.0 / Rs;

    objects[2].mass = M;
    objects[2].radius = Rs;
}
updateBHParams(4.3e6, 0.0, 0.0); // 4.3e6 solar masses is Sag A*'s mass

const cam = {
    radius: 6.34194e10,
    azimuth: 0.4,
    elevation: 1.25,
    minR: 1e10,
    maxR: 1e12,
    orbitSpd: 0.007,
    zoomSpd: 2.2e9,
    dragging: false,
    rightDown: false,
    lastX: 0,
    lastY: 0,
    pos() {
        const el = Math.max(0.01, Math.min(Math.PI - 0.01, this.elevation));
        return [
            this.radius * Math.sin(el) * Math.cos(this.azimuth),
            this.radius * Math.cos(el),
            this.radius * Math.sin(el) * Math.sin(this.azimuth),
        ];
    },
};

let showAccretionDisk = true;
let bendLight = true;
let frameCount = 0,
    fps = 0,
    fpsTimer = 0;

const canvas = document.getElementById("c");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
const gl = canvas.getContext("webgl2");
if (!gl) {
    alert("WebGL 2 is required (update your browser).");
}

let RT_W = 150, RT_H = 150;

// === !!! Shaders !!! ===
const RT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;

uniform float time;
uniform bool showDisk;
uniform bool bendLight;

uniform vec3 camPos;
uniform vec3 camRight;
uniform vec3 camUp;
uniform vec3 camFwd;
uniform float tanHalfFov;
uniform float aspect;
uniform int numObjects;
uniform vec4 objPR[8];
uniform vec4 objCol[8];
uniform float diskR1;
uniform float diskR2;

// Black hole constants
uniform float BH_M;
uniform float BH_A;
uniform float BH_Q;

const float Rs = 1.0;

float A2;
float Q2;
float BH_RPLUS;

const float ESCAPE_R = 85.0;
const int STEPS = 2000;

// Kerr Metric
// y is spin axis: theta measured from +y & phi = atan(z, x)
// equatorial plane: y = 0 (theta = pi/2)
// Boyer-Lindquist Coords: pos = (t, r, theta, phi), vel = d/dlambda of same

struct Metric    { float tt, rr, qq, pp, tp; };   // covariant g_uv
struct InvMetric { float tt, rr, qq, pp, tp; };   // contravariant g^uv

Metric kerrMetric(float r, float theta) {
    float costheta = cos(theta), sintheta = sin(theta);
    float sintheta2 = max(sintheta*sintheta, 1e-10);
    float Sigma = r*r + A2*costheta*costheta;
    float Delta = r*r - Rs*r + A2 + Q2;

    float charge_term = Rs*r - Q2; // $2Mr - Q^2$

    Metric g;
    g.tt = -(1. - charge_term / Sigma);
    g.rr = Sigma/Delta;
    g.qq = Sigma;
    g.tp = -charge_term * BH_A * sintheta2 / Sigma;
    g.pp = sintheta2 * (r*r + A2 + charge_term * A2 * sintheta2 /  Sigma);
    return g;
}

InvMetric kerrInvMetric(float r, float theta) {
    float costheta = cos(theta), sintheta = sin(theta);
    float sintheta2 = max(sintheta*sintheta, 1e-10);
    float Sigma = r*r + A2*costheta*costheta;
    float Delta = r*r - Rs*r + A2 + Q2;

    float r2a2 = r*r + A2;
    float SD = Sigma * Delta;
    InvMetric gi;
    gi.tt = -(r2a2*r2a2 - Delta*A2*sintheta2) / SD;
    gi.rr = Delta/Sigma;
    gi.qq = 1./Sigma;
    gi.pp = (Delta - A2*sintheta2) / (SD * sintheta2);
    gi.tp = -Rs * r * BH_A / SD;
    return gi;
}

// Geodesic Acceleration
// Kerr has two Killing vectors (d/dt, d/dphi) => only d_r and d_theta
// derivatives of the metric are non-zero, so exploit this to only evaluate:
// r +- h & theta+- h instead of 8 values
vec4 kerrAccel(vec4 pos, vec4 vel) {
    float r = pos.y;
    float theta = pos.z;
    float vt = vel.x, vr = vel.y, vq = vel.z, vp = vel.w;

    float hr = max(abs(r), 0.01) * 1e-3;
    float hq = 1e-3;

    Metric gRp = kerrMetric(r + hr, theta);
    Metric gRm = kerrMetric(r - hr, theta);
    Metric gQp = kerrMetric(r, theta + hq);
    Metric gQm = kerrMetric(r, theta - hq);

    float i2hr = 0.5 / hr;
    float i2hq = 0.5 / hq;

    // Central-difference metric derivatives
    float dR_tt = (gRp.tt - gRm.tt) * i2hr;
    float dR_rr = (gRp.rr - gRm.rr) * i2hr;
    float dR_qq = (gRp.qq - gRm.qq) * i2hr;
    float dR_pp = (gRp.pp - gRm.pp) * i2hr;
    float dR_tp = (gRp.tp - gRm.tp) * i2hr;

    float dQ_tt = (gQp.tt - gQm.tt) * i2hq;
    float dQ_rr = (gQp.rr - gQm.rr) * i2hq;
    float dQ_qq = (gQp.qq - gQm.qq) * i2hq;
    float dQ_pp = (gQp.pp - gQm.pp) * i2hq;
    float dQ_tp = (gQp.tp - gQm.tp) * i2hq;

    // Σ_{a in {r,theta}, b} d_a g_{s,b} v^a v^b  summed over all b
    // (g_{s,b} non-zero only for certain b depending on s)

    // Total d_r and d_theta quadratic forms (for the -d_sigma term)
    float dr_sum = dR_tt*vt*vt + 2.0*dR_tp*vt*vp
                 + dR_rr*vr*vr + dR_qq*vq*vq + dR_pp*vp*vp;
    float dq_sum = dQ_tt*vt*vt + 2.0*dQ_tp*vt*vp
                 + dQ_rr*vr*vr + dQ_qq*vq*vq + dQ_pp*vp*vp;

    // F_t  (sigma=t,  no d_t term)
    float Ft = 2.0*(dR_tt*vr*vt + dR_tp*vr*vp
                  + dQ_tt*vq*vt + dQ_tp*vq*vp);

    // F_r  (sigma=r,  g_{r,b} nonzero only for b=r)
    float Fr = 2.0*(dR_rr*vr + dQ_rr*vq)*vr - dr_sum;

    // F_theta  (sigma=theta, g_{q,b} nonzero only for b=theta)
    float Fq = 2.0*(dR_qq*vr + dQ_qq*vq)*vq - dq_sum;

    // F_phi  (sigma=phi, no d_phi term)
    float Fp = 2.0*(dR_tp*vr*vt + dR_pp*vr*vp
                  + dQ_tp*vq*vt + dQ_pp*vq*vp);

    InvMetric gi = kerrInvMetric(r, theta);

    return vec4(
        -0.5*(gi.tt*Ft + gi.tp*Fp), // a^t
        -0.5* gi.rr*Fr,             // a^r
        -0.5* gi.qq*Fq,             // a^theta
        -0.5*(gi.tp*Ft + gi.pp*Fp)  // a^phi
    );
}

float dynStep(vec4 pos, vec4 vel) {
    vec4 av = abs(vel);
    float div = max(max(av.x, av.y), max(av.z, av.w));
    div = max(div, 1e-6);
    float r = abs(pos.y);
    if (r < 1.5*Rs) return 0.005 / div;
    if (r < 5.0*Rs) return 0.012 / div;
                    return 0.05  / div;
}

struct Ray { vec4 pos; vec4 vel; float x; float y; float z; vec3 flatDir; };

void syncCart(inout Ray ray) {
    float r   = ray.pos.y, theta = ray.pos.z, phi = ray.pos.w;
    float ct  = cos(theta), st = sin(theta), cp = cos(phi), sp = sin(phi);
    ray.x = r*st*cp;   // x = r sinθ cosφ
    ray.y = r*ct;      // y = r cosθ
    ray.z = r*st*sp;   // z = r sinθ sinφ
}

Ray initRay(vec3 pos, vec3 dir) {
    Ray ray;
    float r   = length(pos);
    float ct  = clamp(pos.y / r, -1.0, 1.0);
    float th  = acos(ct);
    float ph  = atan(pos.z, pos.x);
    float st  = sin(th), cp = cos(ph), sp = sin(ph);

    // Decompose Cartesian direction into BL coordinate velocities.
    // Basis vectors (y-pole):
    //   r-hat = ( sinθ cosφ,  cosθ,  sinθ sinφ )
    //   θ-hat = ( cosθ cosφ, -sinθ,  cosθ sinφ )
    //   φ-hat = (     -sinφ,     0,       cosφ )
    float dr     = st*cp*dir.x + ct*dir.y + st*sp*dir.z;
    float dtheta = (ct*cp*dir.x - st*dir.y + ct*sp*dir.z) / max(r, 0.001);
    float dphi   = (-sp*dir.x + cp*dir.z) / max(r*st, 0.001);

    // Null initialisation: g_{uv} v^u v^v = 0  =>  A dt^2 + 2B dt + C = 0
    // A = g_tt < 0 outside ergosphere, so the positive root (forward time) is:
    //   dt = (-B - sqrt(B^2 - A*C)) / A
    Metric g = kerrMetric(r, th);
    float A   = g.tt;
    float B   = g.tp * dphi;
    float Cq  = g.rr*dr*dr + g.qq*dtheta*dtheta + g.pp*dphi*dphi;
    float dt  = (-B - sqrt(max(B*B - A*Cq, 0.0))) / A;

    ray.pos = vec4(0.0, r, th, ph);
    ray.vel = vec4(dt, dr, dtheta, dphi);
    ray.x = pos.x; ray.y = pos.y; ray.z = pos.z;
    ray.flatDir = dir;
    return ray;
}

void stepRay(inout Ray ray) {
    if (!bendLight) {
        float dL = 0.04;
        ray.x += ray.flatDir.x * dL;
        ray.y += ray.flatDir.y * dL;
        ray.z += ray.flatDir.z * dL;
        float r  = length(vec3(ray.x, ray.y, ray.z));
        float th = acos(clamp(ray.y / max(r, 0.001), -1.0, 1.0));
        float ph = atan(ray.z, ray.x);
        ray.pos  = vec4(ray.pos.x, r, th, ph);
        return;
    }
    float dL  = dynStep(ray.pos, ray.vel);
    vec4  acc = kerrAccel(ray.pos, ray.vel);
    ray.vel  += acc * dL;
    ray.pos  += ray.vel * dL;
    syncCart(ray);
}

bool diskCross(vec3 a, vec3 b) {
    if (a.y * b.y >= 0.0) return false;
    float r = length(vec2(b.x, b.z));
    return r >= diskR1 && r <= diskR2;
}

float hash(float n)  { return fract(sin(n) * 43758.5453); }
float hash3(vec3 v)  { return hash(v.x*127.1 + v.y*311.7 + v.z*74.7); }

void main() {    
    A2 = BH_A * BH_A;
    Q2 = BH_Q * BH_Q;
    BH_RPLUS = BH_M + sqrt(max(BH_M*BH_M - A2 - Q2, 0.0));
    float u   = (2.0*vUV.x - 1.0) * aspect * tanHalfFov;
    float v   = (1.0 - 2.0*vUV.y) * tanHalfFov;
    vec3  dir = normalize(u*camRight - v*camUp + camFwd);

    Ray  ray   = initRay(camPos, dir);
    vec3 prevP = vec3(ray.x, ray.y, ray.z);

    bool hitBH = false, hitDisk = false, hitObj = false;
    vec4 objHitCol = vec4(0.0);
    vec3 hitCentre = vec3(0.0);

    for (int i = 0; i < STEPS; i++) {
        if (ray.pos.y <= BH_RPLUS * 1.02) { hitBH = true; break; }

        stepRay(ray);
        vec3 newP = vec3(ray.x, ray.y, ray.z);

        if (showDisk && diskCross(prevP, newP)) { hitDisk = true; break; }

        for (int j = 0; j < numObjects; j++) {
            if (distance(newP, objPR[j].xyz) <= objPR[j].w) {
                objHitCol = objCol[j];
                hitCentre = objPR[j].xyz;
                hitObj    = true;
                break;
            }
        }
        if (hitObj) break;
        prevP = newP;
        if (ray.pos.y > ESCAPE_R) break;
    }

    vec4 color;

    if (hitDisk) {
        float rn   = length(vec2(ray.x, ray.z)) / diskR2;
        float glow  = exp(-rn * 5.0);
        float bloom = pow(max(1.0 - rn, 0.0), 8.0) * 3.0;
        vec3 inner = vec3(0.6775, 0.199019607843, 0.0049019607825);  // orange
        vec3 outer = vec3(1.0,  0.90, 0.35);   // yellow
        vec3 dc     = mix(inner, outer, rn);
        color = vec4(dc * (0.85 + 2.0*glow) + inner * bloom, 1.0);
    } else if (hitBH) {
        color = vec4(0.0, 0.0, 0.0, 1.0);

    } else if (hitObj) {
        vec3  P = vec3(ray.x, ray.y, ray.z);
        vec3  N = normalize(P - hitCentre);
        vec3  V = normalize(camPos - P);
        float d = max(dot(N, V), 0.0);
        color = vec4(objHitCol.rgb * (0.1 + 0.9*d), 1.0);

    } else {
        // Starfield 
        // vec3 d3=dir*120.; vec3 cell=floor(d3);
        // float h1=hash3(cell); float h3=hash(h1*537.9);
        // float star=pow(max(h3-0.955,0.)*21.,2.5);
        // vec3 tint=mix(vec3(0.7,0.8,1.),vec3(1.,0.85,0.6), hash(h1*73.));
        // color=vec4(tint*star, 1.);
        color = vec4(0.0, 0.0, 0.0, 0.9);
    }

    frag = color;
}
`;

const QUAD_VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ gl_Position=vec4(aPos,0.,1.); vUV=aUV; }`;

const BLIT_FS = `#version 300 es
precision mediump float;
in vec2 vUV;
out vec4 frag;
uniform sampler2D uTex;
void main(){
  frag = texture(uTex, vUV);
}`;

const GRID_VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uVP;
uniform float bhRs;
uniform float bhA;

void main() {
    float r     = length(aPos.xz);
    float rNorm = max(r / bhRs, 1.0);
    float angle =  bhA / (2.0 * rNorm * rNorm * rNorm);

    float ca = cos(angle), sa = sin(angle);
    vec3 dragged = vec3(
        aPos.x * ca - aPos.z * sa,
        aPos.y,
        aPos.x * sa + aPos.z * ca
    );

    gl_Position = uVP * vec4(dragged, 1.0);
}`;

const GRID_FS = `#version 300 es
precision mediump float;
out vec4 frag;
// void main(){ frag=vec4(.08,.49,.30,.90);
void main(){ frag=vec4(1.,1.,1.,.3);

}`;

function compSh(src, type) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(s), "\n---\n", src.slice(0, 300));
    return s;
}
function mkProg(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compSh(vs, gl.VERTEX_SHADER));
    gl.attachShader(p, compSh(fs, gl.FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(p));
    return p;
}

const rtProg = mkProg(QUAD_VS, RT_FS);
const blitProg = mkProg(QUAD_VS, BLIT_FS);
const gridProg = mkProg(GRID_VS, GRID_FS);

const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const qv = new Float32Array([
    -1, 1, 0, 1, -1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1,
]);
const quadVBO = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
gl.bufferData(gl.ARRAY_BUFFER, qv, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

const rtFBO = gl.createFramebuffer();
const rtTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, rtTex);
function resizeRT(w, h) {
    RT_W = w; RT_H = h;
    gl.bindTexture(gl.TEXTURE_2D, rtTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, RT_W, RT_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, rtFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rtTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
resizeRT(RT_W, RT_H);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.bindFramebuffer(gl.FRAMEBUFFER, rtFBO);
gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    rtTex,
    0,
);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);

const gridVAO = gl.createVertexArray();
const gridVBO = gl.createBuffer();
const gridEBO = gl.createBuffer();
gl.bindVertexArray(gridVAO);
gl.bindBuffer(gl.ARRAY_BUFFER, gridVBO);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gridEBO);
let gridIdxCount = 0;

function generateGrid() {
    const N = 24,
        spacing = 1e10;
    const verts = [],
        idxs = [];
    for (let iz = 0; iz <= N; iz++) {
        for (let ix = 0; ix <= N; ix++) {
            const wx = (ix - N / 2) * spacing,
                wz = (iz - N / 2) * spacing;
            let y = 0;
            for (const o of objects) {
                const rs = (2 * G * o.mass) / (c * c);
                const dx = wx - o.pos[0],
                    dz = wz - o.pos[2];
                const dist = Math.sqrt(dx * dx + dz * dz);
                y += dist > rs ? 2 * Math.sqrt(rs * (dist - rs)) - 3e10 : 2 * rs - 3e10;
            }
            verts.push(wx, y, wz);
        }
    }
    for (let iz = 0; iz < N; iz++)
        for (let ix = 0; ix < N; ix++) {
            const i = iz * (N + 1) + ix;
            idxs.push(i, i + 1, i, i + N + 1);
        }
    gl.bindVertexArray(gridVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, gridVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gridEBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idxs), gl.STATIC_DRAW);
    gridIdxCount = idxs.length;
}

function mat4() {
    return new Float32Array(16);
}
function identity(m) {
    m.fill(0);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
}

function perspective(out, fovY, asp, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    out.fill(0);
    out[0] = f / asp;
    out[5] = f;
    out[10] = (near + far) / (near - far);
    out[11] = -1;
    out[14] = (2 * near * far) / (near - far);
    return out;
}

function lookAt(out, eye, cen, up) {
    let fx = cen[0] - eye[0],
        fy = cen[1] - eye[1],
        fz = cen[2] - eye[2];
    let fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl;
    fy /= fl;
    fz /= fl;
    let rx = fy * up[2] - fz * up[1],
        ry = fz * up[0] - fx * up[2],
        rz = fx * up[1] - fy * up[0];
    let rl = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rl;
    ry /= rl;
    rz /= rl;
    const ux = ry * fz - rz * fy,
        uy = rz * fx - rx * fz,
        uz = rx * fy - ry * fx;
    out[0] = rx;
    out[1] = ux;
    out[2] = -fx;
    out[3] = 0;
    out[4] = ry;
    out[5] = uy;
    out[6] = -fy;
    out[7] = 0;
    out[8] = rz;
    out[9] = uz;
    out[10] = -fz;
    out[11] = 0;
    out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
    out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
    out[15] = 1;
    return out;
}

function mulMat(out, a, b) {
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[j * 4 + k];
            out[j * 4 + i] = s;
        }
    return out;
}

function norm3(v) {
    const l = Math.hypot(...v);
    return v.map((x) => x / l);
}
function cross3(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}
function sub3(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

// Input variables
const rtU = {
    camPos: gl.getUniformLocation(rtProg, "camPos"),
    camRight: gl.getUniformLocation(rtProg, "camRight"),
    camUp: gl.getUniformLocation(rtProg, "camUp"),
    camFwd: gl.getUniformLocation(rtProg, "camFwd"),
    tanHalfFov: gl.getUniformLocation(rtProg, "tanHalfFov"),
    aspect: gl.getUniformLocation(rtProg, "aspect"),
    numObjects: gl.getUniformLocation(rtProg, "numObjects"),
    objPR: gl.getUniformLocation(rtProg, "objPR"),
    objCol: gl.getUniformLocation(rtProg, "objCol"),
    diskR1: gl.getUniformLocation(rtProg, "diskR1"),
    diskR2: gl.getUniformLocation(rtProg, "diskR2"),
    time: gl.getUniformLocation(rtProg, "time"),
    showDisk: gl.getUniformLocation(rtProg, "showDisk"),
    bendLight: gl.getUniformLocation(rtProg, "bendLight"),
    bhM: gl.getUniformLocation(rtProg, "BH_M"),
    bhA: gl.getUniformLocation(rtProg, "BH_A"),
    bhQ: gl.getUniformLocation(rtProg, "BH_Q"),
};
const blitU = { tex: gl.getUniformLocation(blitProg, "uTex") };
const gridU = {
    vp: gl.getUniformLocation(gridProg, "uVP"),
    bhRS: gl.getUniformLocation(gridProg, "bhRS"),
    bhA: gl.getUniformLocation(gridProg, "bhA"),
};

canvas.addEventListener("mousedown", (e) => {
    e.preventDefault();
    if (e.button === 0) {
        cam.dragging = true;
    }
    cam.lastX = e.clientX;
    cam.lastY = e.clientY;
});
canvas.addEventListener("mouseup", (e) => {
    if (e.button === 0) cam.dragging = false;
});
canvas.addEventListener("mousemove", (e) => {
    if (!cam.dragging) return;
    cam.azimuth += (e.clientX - cam.lastX) * cam.orbitSpd;
    cam.elevation -= (e.clientY - cam.lastY) * cam.orbitSpd;
    cam.elevation = Math.max(0.01, Math.min(Math.PI - 0.01, cam.elevation));
    cam.lastX = e.clientX;
    cam.lastY = e.clientY;
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener(
    "wheel",
    (e) => {
        cam.radius = Math.max(
            cam.minR,
            Math.min(cam.maxR, cam.radius * (1 + e.deltaY * 0.00015)),
        );
    },
    { passive: true },
);

window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

// !!! Render
const proj = mat4();
const view = mat4();
const viewProj = mat4();

let lastT = 0;

function render(t) {
    requestAnimationFrame(render);
    const dt = Math.min((t - lastT) * 0.001, 0.05);
    lastT = t;

    frameCount++;
    fpsTimer += dt;
    if (fpsTimer > 0.5) {
        fps = Math.round(frameCount / fpsTimer);
        frameCount = 0;
        fpsTimer = 0;
    }

    const pos = cam.pos();
    const target = [0, 0, 0];
    const up = [0, 1, 0];
    const fwd = norm3(sub3(target, pos));
    const right = norm3(cross3(fwd, up));
    const camUp = cross3(right, fwd);
    const tanHFov = Math.tan(Math.PI / 6); // 60° FOV
    const aspect = canvas.width / canvas.height;

    const n = objects.length;
    const prArr = new Float32Array(n * 4);
    const colArr = new Float32Array(n * 4);
    objects.forEach((o, i) => {
        prArr.set(
            [o.pos[0] * scale, o.pos[1] * scale, o.pos[2] * scale, o.radius * scale],
            i * 4,
        );
        colArr.set(o.color, i * 4);
    });

    // Raytrace pass -> FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, rtFBO);
    gl.viewport(0, 0, RT_W, RT_H);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(rtProg);

    gl.uniform3fv(
        rtU.camPos,
        pos.map((x) => x * scale),
    );
    gl.uniform3fv(rtU.camRight, right);
    gl.uniform3fv(rtU.camUp, camUp);
    gl.uniform3fv(rtU.camFwd, fwd);
    gl.uniform1f(rtU.tanHalfFov, tanHFov);
    gl.uniform1f(rtU.aspect, aspect);
    gl.uniform1i(rtU.numObjects, n);
    gl.uniform4fv(rtU.objPR, prArr);
    gl.uniform4fv(rtU.objCol, colArr);
    gl.uniform1f(rtU.diskR1, Rs * 2.2 * scale);
    gl.uniform1f(rtU.diskR2, Rs * 5.2 * scale);
    gl.uniform1f(rtU.time, t * 0.001);
    gl.uniform1i(rtU.showDisk, showAccretionDisk ? 1 : 0);
    gl.uniform1f(rtU.bhM, 0.5);
    gl.uniform1f(rtU.bhA, aStar);
    gl.uniform1f(rtU.bhQ, QStar);
    gl.uniform1i(rtU.bendLight, bendLight ? 1 : 0);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);



    // Blit to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(blitProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, rtTex);
    gl.uniform1i(blitU.tex, 0);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    generateGrid();

    const fullAspect = canvas.width / canvas.height;
    perspective(proj, Math.PI / 3, fullAspect, 1e9, 1e14);
    lookAt(view, pos, target, up);
    mulMat(viewProj, proj, view);

    gl.useProgram(gridProg);
    gl.uniformMatrix4fv(gridU.vp, false, viewProj);
    gl.uniform1f(gridU.bhRS, Rs);
    gl.uniform1f(gridU.bhA, aStar);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(gridVAO);
    gl.drawElements(gl.LINES, gridIdxCount, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);

    // HUD
    const rNorm = cam.radius * scale;
    const rStr = (cam.radius / 1e9).toFixed(2);
    const pSI = pos.map((x) => (x / 1e9).toFixed(1));
    const star0d = Math.sqrt(objects[0].pos.reduce((a, x) => a + x * x, 0)) / 1e9;
    const star1d = Math.sqrt(objects[1].pos.reduce((a, x) => a + x * x, 0)) / 1e9;
}

requestAnimationFrame(render);

const resSelect = document.getElementById('resSelect');
const chkDisk = document.getElementById('chkDisk');
const chkBend = document.getElementById('chkBend');
const inMass = document.getElementById('inMass');
const inSpin = document.getElementById('inSpin');
const inCharge = document.getElementById('inCharge');
const btnApply = document.getElementById('btnApply');
const validMsg = document.getElementById('validMsg');

chkDisk.addEventListener('change', () => { showAccretionDisk = chkDisk.checked; });
chkBend.addEventListener('change', () => { bendLight = chkBend.checked; });

resSelect.addEventListener('change', () => {
    const res = parseInt(resSelect.value);
    resizeRT(res, res);
});

btnApply.addEventListener('click', () => {
    validMsg.textContent = '';

    const massSolar = parseFloat(inMass.value);
    const aIn = parseFloat(inSpin.value);
    const qIn = parseFloat(inCharge.value);

    if (isNaN(massSolar) || massSolar <= 0) {
        validMsg.textContent = 'Mass must be a positive number.'; return;
    }
    if (isNaN(aIn) || aIn < 0 || aIn >= 1) {
        validMsg.textContent = 'Spin a* must be in [0, 1).'; return;
    }
    if (isNaN(qIn) || qIn < 0 || qIn >= 1) {
        validMsg.textContent = 'Charge Q* must be in [0, 1).'; return;
    }
    if (aIn * aIn + qIn * qIn > 1.0) {
        validMsg.textContent =
            `Invalid: a*² + Q*² = ${(aIn * aIn + qIn * qIn).toFixed(4)} > 1.  ` +
            `Reduce spin or charge to keep a black hole.`;
        return;
    }

    updateBHParams(massSolar, aIn, qIn);

    const res = parseInt(resSelect.value);
    resizeRT(res, res);

    validMsg.style.color = '#8f8';
    validMsg.textContent = `✓ Applied — Rs = ${Rs.toExponential(3)} m`;
    setTimeout(() => { validMsg.textContent = ''; validMsg.style.color = '#f88'; }, 3000);
});