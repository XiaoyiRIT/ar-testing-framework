/*
 * HelloSceneformActivity with AR_OP real-time logs for operation success metrics.
 * - AndroidX AppCompat
 * - View-level gesture detection (works even when finger is off the model)
 * - Separated `pinch` and `rotate` logs with dominance rule to avoid double fire
 * - Place success confirmed by consecutive TRACKING frames
 */
package com.google.ar.sceneform.samples.hellosceneform;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.ar.core.Anchor;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import org.json.JSONObject;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class HelloSceneformActivity extends AppCompatActivity {
    private static final String TAG = HelloSceneformActivity.class.getSimpleName();
    private static final double MIN_OPENGL_VERSION = 3.0;
    private static final String OP_TAG = "AR_OP"; // Logcat tag consumed by the test tool

    // Success thresholds (tunable per device)
    private static final float EPS_T_M = 0.02f; // drag: 2 cm
    private static final float EPS_R_DEG = 3f;  // rotate: 3 degrees (more sensitive)
    private static final float EPS_S = 0.02f;   // pinch: 2% scale change (more sensitive)

    // Place confirmation frames while TRACKING
    private static final int PLACE_CONFIRM_FRAMES = 3;

    private ArFragment arFragment;
    private ModelRenderable andyRenderable;

    // View-level detectors (work anywhere on the view, not only on the node)
    private ScaleGestureDetector scaleDetector;
    private final RotationAccumulator rotationAccumulator = new RotationAccumulator();

    // One active gesture session bound to the currently selected node
    private GestureSession currentSession = null;

    // Pending anchors waiting for TRACKING confirmation
    private final Map<Anchor, Integer> pendingAnchors = new HashMap<>();

    @Override
    @SuppressWarnings({"AndroidApiChecker", "FutureReturnValueIgnored"})
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!checkIsSupportedDeviceOrFinish(this)) return;

        setContentView(R.layout.activity_ux);
        arFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);

        ModelRenderable.builder()
                .setSource(this, R.raw.andy)
                .build()
                .thenAccept(r -> andyRenderable = r)
                .exceptionally(throwable -> {
                    Toast.makeText(this, "Unable to load andy renderable", Toast.LENGTH_LONG).show();
                    return null;
                });

        // Confirm place success by consecutive TRACKING frames
        arFragment.getArSceneView().getScene().addOnUpdateListener(frameTime -> {
            if (pendingAnchors.isEmpty()) return;
            Iterator<Map.Entry<Anchor, Integer>> it = pendingAnchors.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry<Anchor, Integer> e = it.next();
                Anchor a = e.getKey();
                if (a.getTrackingState() == TrackingState.TRACKING) {
                    int c = e.getValue() + 1;
                    if (c >= PLACE_CONFIRM_FRAMES) {
                        logJson("place_ok", true, kv("anchor_pose", vec3(a.getPose())));
                        it.remove();
                    } else {
                        e.setValue(c);
                    }
                } else if (a.getTrackingState() == TrackingState.STOPPED) {
                    logJson("place_fail", false, kv("reason", "anchor_stopped"));
                    it.remove();
                }
            }
        });

        // View-level gesture detectors
        scaleDetector = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector detector) {
                if (currentSession != null) currentSession.accumScale *= detector.getScaleFactor();
                return false; // let Sceneform's UX also handle
            }
        });

        try {
            arFragment.getArSceneView().setOnTouchListener((v, ev) -> {
                // Feed detectors regardless of hit-test; Sceneform will still process its own UX.
                scaleDetector.onTouchEvent(ev);
                rotationAccumulator.onTouchEvent(ev);

                switch (ev.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN: {
                        TransformableNode sel = (TransformableNode) arFragment.getTransformationSystem().getSelectedNode();
                        if (sel == null) return false;
                        currentSession = (sel != null) ? new GestureSession(sel) : null;
                        break;
                    }
                    case MotionEvent.ACTION_POINTER_DOWN: {
                        if (currentSession == null) {
                            TransformableNode sel = (TransformableNode) arFragment.getTransformationSystem().getSelectedNode();
                            if (sel == null) return false;
                            currentSession = (sel != null) ? new GestureSession(sel) : null;
                        }
                        break;
                    }
                    case MotionEvent.ACTION_MOVE: {
                        if (currentSession != null) {
                            currentSession.maxPointers = Math.max(currentSession.maxPointers, ev.getPointerCount());
                            currentSession.accumRotateDeg += rotationAccumulator.consumeDeltaDegrees();
                        }
                        break;
                    }
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_POINTER_UP:
                    case MotionEvent.ACTION_CANCEL: {
                        if (currentSession != null) {
                            TransformableNode node = currentSession.node;
                            // Compare LOCAL transforms (robust to camera motion)
                            Vector3 pos1 = node.getLocalPosition();
                            Quaternion rot1 = node.getLocalRotation();
                            Vector3 scale1 = node.getLocalScale();

                            float dTrans = Vector3.subtract(pos1, currentSession.pos0).length();
                            float dYawLocal = Math.abs(normalizeDeg(yawDeg(rot1) - yawDeg(currentSession.rot0)));
                            float s0 = currentSession.scale0.x, s1 = scale1.x;
                            float dScaleAbs = Math.abs((s1 - s0) / Math.max(1e-6f, s0));

                            if (currentSession.maxPointers <= 1) {
                                boolean dragOk = dTrans >= EPS_T_M;
                                logJson("drag", dragOk, kv("dTrans_m", dTrans));
                            } else {
                                boolean pinchOk = dScaleAbs >= EPS_S || Math.abs(currentSession.accumScale - 1f) >= EPS_S;
                                boolean rotateOk = Math.abs(currentSession.accumRotateDeg) >= EPS_R_DEG || dYawLocal >= EPS_R_DEG;

                                if (pinchOk && !rotateOk) {
                                    logJson("pinch", true,
                                            kv("scale_factor", currentSession.accumScale),
                                            kv("dScale_abs", dScaleAbs));
                                } else if (rotateOk && !pinchOk) {
                                    logJson("rotate", true,
                                            kv("dYaw_deg_screen", currentSession.accumRotateDeg),
                                            kv("dYaw_local_deg", dYawLocal));
                                } else if (pinchOk && rotateOk) {
                                    // Dominance rule to avoid double fire except when both are strong
                                    float pinchScore = Math.abs(currentSession.accumScale - 1f) / Math.max(1e-6f, EPS_S);
                                    float rotScore   = Math.abs(currentSession.accumRotateDeg) / Math.max(1e-6f, EPS_R_DEG);
                                    if (pinchScore > rotScore * 1.4f) {
                                        logJson("pinch", true,
                                                kv("scale_factor", currentSession.accumScale),
                                                kv("dScale_abs", dScaleAbs),
                                                kv("coupled", true));
                                    } else if (rotScore > pinchScore * 1.4f) {
                                        logJson("rotate", true,
                                                kv("dYaw_deg_screen", currentSession.accumRotateDeg),
                                                kv("dYaw_local_deg", dYawLocal),
                                                kv("coupled", true));
                                    } else {
                                        // both strong -> emit both once
                                        logJson("pinch", true,
                                                kv("scale_factor", currentSession.accumScale),
                                                kv("dScale_abs", dScaleAbs),
                                                kv("coupled", true));
                                        logJson("rotate", true,
                                                kv("dYaw_deg_screen", currentSession.accumRotateDeg),
                                                kv("dYaw_local_deg", dYawLocal),
                                                kv("coupled", true));
                                    }
                                }
                            }
                            currentSession = null;
                            rotationAccumulator.reset();
                        }
                        break;
                    }
                }
                return false; // don't consume; keep Sceneform UX
            });
        } catch (Throwable t) {
            Log.e("AR_OP", "gesture_runtime_error", t);
        }

        // Tap to place an object and select it
        arFragment.setOnTapArPlaneListener((HitResult hitResult, Plane plane, MotionEvent motionEvent) -> {
            if (andyRenderable == null) return;

            Anchor anchor = hitResult.createAnchor();
            AnchorNode anchorNode = new AnchorNode(anchor);
            anchorNode.setParent(arFragment.getArSceneView().getScene());

            TransformableNode andy = new TransformableNode(arFragment.getTransformationSystem());
            andy.setParent(anchorNode);
            andy.setRenderable(andyRenderable);
            andy.select();

            logJson("place_start", true, kv("anchor_pose", vec3(anchor.getPose())));
            pendingAnchors.put(anchor, 0);
        });
    }

    // -------------------- Gesture session & rotation accumulator --------------------
    private static class GestureSession {
        final TransformableNode node;
        final Vector3 pos0;
        final Quaternion rot0;
        final Vector3 scale0;
        int maxPointers = 0;
        float accumScale = 1.0f;    // from ScaleGestureDetector
        float accumRotateDeg = 0f;  // from RotationAccumulator
        GestureSession(TransformableNode n) {
            node = n;
            pos0 = n.getLocalPosition();
            rot0 = n.getLocalRotation();
            scale0 = n.getLocalScale();
        }
    }

    // Accumulate rotation (screen space) between two pointers
    private static class RotationAccumulator {
        private boolean active = false;
        private float lastAngle = 0f;
        private float deltaSum = 0f;

        void reset(){ active=false; lastAngle=0f; deltaSum=0f; }

        void onTouchEvent(MotionEvent ev){
            if (ev.getPointerCount() < 2) { active=false; return; }
            float x0 = ev.getX(0), y0 = ev.getY(0);
            float x1 = ev.getX(1), y1 = ev.getY(1);
            float ang = (float)Math.toDegrees(Math.atan2(y1 - y0, x1 - x0));
            if (!active) { active = true; lastAngle = ang; return; }
            float d = normalizeDeg(ang - lastAngle);
            deltaSum += d;
            lastAngle = ang;
        }

        float consumeDeltaDegrees(){ float d = deltaSum; deltaSum = 0f; return d; }
    }

    // -------------------- Logging helpers --------------------
    private static Object[] kv(String k, Object v) { return new Object[]{k, v}; }

    private static void logJson(String kind, boolean ok, Object[]... kvs) {
        try {
            JSONObject j = new JSONObject();
            j.put("kind", kind);
            j.put("ok", ok);
            j.put("ts_wall", System.currentTimeMillis());
            for (Object[] kv : kvs) j.put((String) kv[0], kv[1]);
            Log.d(OP_TAG, j.toString());
        } catch (Exception ignore) {}
    }

    // Vector helpers
    private static String vec3(Pose pose) {
        float[] t = pose.getTranslation();
        return t[0] + "," + t[1] + "," + t[2];
    }

    // Math helpers
    private static float yawDeg(Quaternion q) {
        double t3 = 2.0 * (q.w * q.y + q.z * q.x);
        double t4 = 1.0 - 2.0 * (q.y * q.y + q.x * q.x);
        return (float) Math.toDegrees(Math.atan2(t3, t4));
    }
    private static float normalizeDeg(float a) {
        while (a > 180) a -= 360; while (a < -180) a += 360; return a;
    }

    // -------------------- Stock sample helpers (unchanged) --------------------
    public static boolean checkIsSupportedDeviceOrFinish(final Activity activity) {
        if (Build.VERSION.SDK_INT < VERSION_CODES.N) {
            Log.e(TAG, "Sceneform requires Android N or later");
            Toast.makeText(activity, "Sceneform requires Android N or later", Toast.LENGTH_LONG).show();
            activity.finish();
            return false;
        }
        String openGlVersionString =
                ((ActivityManager) activity.getSystemService(Context.ACTIVITY_SERVICE))
                        .getDeviceConfigurationInfo()
                        .getGlEsVersion();
        if (Double.parseDouble(openGlVersionString) < MIN_OPENGL_VERSION) {
            Log.e(TAG, "Sceneform requires OpenGL ES 3.0 or later");
            Toast.makeText(activity, "Sceneform requires OpenGL ES 3.0 or later", Toast.LENGTH_LONG).show();
            activity.finish();
            return false;
        }
        return true;
    }
}
