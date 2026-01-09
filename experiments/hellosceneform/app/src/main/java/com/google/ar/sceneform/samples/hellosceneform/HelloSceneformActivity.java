/*
 * HelloSceneformActivity — stable logging using Scene.addOnPeekTouchListener (no view-level onTouch)
 * - AndroidX AppCompat
 * - Logcat single-line JSON (TAG=AR_OP)
 * - Every single-finger light tap emits one `tap` event (target: plane|node|empty)
 * - Plane taps also emit `place_*` (start/ok/fail) with the same tap_id
 * - Drag / Pinch / Rotate logging (pinch & rotate independent)
 * - Uses Scene.addOnPeekTouchListener so it never interferes with Sceneform UX
 *
 * This version fixes missing pinch/rotate by running a robust two-finger session
 * that tracks pointer ids, angle and distance deltas across MOVE events, and
 * finalizes when the gesture ends.
 */
package com.google.ar.sceneform.samples.hellosceneform;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.AlertDialog;
import android.content.Context;
import android.os.Build;
import android.os.Build.VERSION_CODES;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ViewConfiguration;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.ar.core.Anchor;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.Scene;
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

    // Thresholds (tunable)
    private static final float EPS_T_M = 0.002f; // drag success: 0.2 cm
    private static final float EPS_S = 0.02f;   // pinch success: 2% scale
    private static final float EPS_R_DEG = 3f;  // rotate success: 3 degrees
    private static final int PLACE_CONFIRM_FRAMES = 3; // consecutive TRACKING frames

    private ArFragment arFragment;
    private ModelRenderable andyRenderable;

    // Tap bookkeeping
    private long tapSeq = 0;             // monotonically increasing tap id
    private long currentTapId = -1;      // id for current DOWN..UP
    private long lastPlaceTapId = -1;    // last tap id that triggered place
    private int tapCount = 0;

    // Tap detection vars
    private float tapDownX = 0f, tapDownY = 0f;
    private long tapDownTs = 0L;
    private int touchSlopPx = 0;

    // Gesture-related vars
    private Handler longPressHandler;
    private Runnable longPressRunnable;
    private boolean wasLongPressed;

    // Drag / transform session snapshots
    private TransformableNode downSelectedNode = null;
    private Vector3 downLocalPos = null;
    private Vector3 downLocalScale = null;
    private Quaternion downLocalRot = null;
    private int maxPointers = 0;
    private boolean downOnSelectedNode = false;

    // Two-finger session (robust, pointer-id aware)
    private static class TwoFingerSession {
        boolean active = false;
        int id0 = -1, id1 = -1;     // pointer ids
        float lastX0, lastY0, lastX1, lastY1;
        float accumScale = 1f;
        float accumRotateDeg = 0f;

        void reset() { active=false; id0=id1=-1; accumScale=1f; accumRotateDeg=0f; }

        // start when two pointers present; choose first two indices
        void start(MotionEvent ev){
            if (ev.getPointerCount() < 2) return;
            active = true;
            id0 = ev.getPointerId(0);
            id1 = ev.getPointerId(1);
            lastX0 = ev.getX(0); lastY0 = ev.getY(0);
            lastX1 = ev.getX(1); lastY1 = ev.getY(1);
            accumScale = 1f; accumRotateDeg = 0f;
        }

        void update(MotionEvent ev){
            if (!active) {
                if (ev.getPointerCount() >= 2) start(ev);
                return;
            }
            int idx0 = ev.findPointerIndex(id0);
            int idx1 = ev.findPointerIndex(id1);
            if (idx0 == -1 || idx1 == -1 || ev.getPointerCount() < 2) { active=false; return; }

            float x0 = ev.getX(idx0), y0 = ev.getY(idx0);
            float x1 = ev.getX(idx1), y1 = ev.getY(idx1);

            // distance & angle last
            float dxPrev = lastX1 - lastX0, dyPrev = lastY1 - lastY0;
            float dxCurr = x1 - x0, dyCurr = y1 - y0;
            float distPrev = (float)Math.hypot(dxPrev, dyPrev);
            float distCurr = (float)Math.hypot(dxCurr, dyCurr);
            if (distPrev > 0.5f && distCurr > 0.5f) {
                float scale = distCurr / distPrev;
                accumScale *= scale;
            }
            float angPrev = (float)Math.toDegrees(Math.atan2(dyPrev, dxPrev));
            float angCurr = (float)Math.toDegrees(Math.atan2(dyCurr, dxCurr));
            float dAng = normalizeDeg(angCurr - angPrev);
            accumRotateDeg += dAng;

            lastX0 = x0; lastY0 = y0; lastX1 = x1; lastY1 = y1;
        }
    }
    private final TwoFingerSession twoF = new TwoFingerSession();

    // Place confirmation
    private final Map<Anchor, Integer> pendingAnchors = new HashMap<>();

    @Override
    @SuppressWarnings({"AndroidApiChecker", "FutureReturnValueIgnored"})
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!checkIsSupportedDeviceOrFinish(this)) return;

        setContentView(R.layout.activity_ux);
        arFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);

        touchSlopPx = ViewConfiguration.get(this).getScaledTouchSlop();
        longPressHandler = new Handler();
        longPressRunnable = () -> {
            wasLongPressed = true;
            logJson("long_press_hold", true, kv("tap_id", currentTapId));
        };

        // Load model
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
                        logJson("place_ok", true, kv("tap_id", lastPlaceTapId), kv("anchor_pose", vec3(a.getPose())));
                        it.remove();
                    } else {
                        e.setValue(c);
                    }
                } else if (a.getTrackingState() == TrackingState.STOPPED) {
                    logJson("place_fail", false, kv("tap_id", lastPlaceTapId), kv("reason", "anchor_stopped"));
                    it.remove();
                }
            }
        });

        // Tap to place an object and select it (this also counts as a tap later)
        arFragment.setOnTapArPlaneListener((HitResult hitResult, Plane plane, MotionEvent motionEvent) -> {
            if (andyRenderable == null) return;

            // Link this place to the most recent tap
            lastPlaceTapId = currentTapId;

            Anchor anchor = hitResult.createAnchor();
            AnchorNode anchorNode = new AnchorNode(anchor);
            anchorNode.setParent(arFragment.getArSceneView().getScene());

            TransformableNode andy = new TransformableNode(arFragment.getTransformationSystem());
            andy.setParent(anchorNode);
            andy.setRenderable(andyRenderable);
            andy.select();

            logJson("place_start", true,
                    kv("tap_id", lastPlaceTapId),
                    kv("anchor_pose", vec3(anchor.getPose())));
            pendingAnchors.put(anchor, 0);
        });

        // --- PeekTouch: observe touches without consuming them (won't break place/UX) ---
        arFragment.getArSceneView().getScene().addOnPeekTouchListener((hit, ev) -> {
            try {
                // Update two-finger session continuously
                twoF.update(ev);

                switch (ev.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN: {
                        currentTapId = ++tapSeq;
                        tapDownX = ev.getX();
                        tapDownY = ev.getY();
                        tapDownTs = SystemClock.uptimeMillis();
                        maxPointers = 1;
                        // snapshot selected node for potential gestures
                        downSelectedNode = (TransformableNode) arFragment.getTransformationSystem().getSelectedNode();
                        downOnSelectedNode = (downSelectedNode != null && hit.getNode() == downSelectedNode);
                        if (downSelectedNode != null) {
                            downLocalPos = downSelectedNode.getLocalPosition();
                            downLocalScale = downSelectedNode.getLocalScale();
                            downLocalRot = downSelectedNode.getLocalRotation();
                        } else {
                            downLocalPos = null; downLocalScale = null; downLocalRot = null;
                        }
                        twoF.reset(); // will start once it sees 2 pointers

                        // Long press
                        wasLongPressed = false;
                        if (hit.getNode() != null && hit.getNode() instanceof TransformableNode) {
                            longPressHandler.postDelayed(longPressRunnable, ViewConfiguration.getLongPressTimeout());
                        }
                        break;
                    }
                    case MotionEvent.ACTION_POINTER_DOWN: {
                        maxPointers = Math.max(maxPointers, ev.getPointerCount());
                        if (ev.getPointerCount() >= 2 && !twoF.active) twoF.start(ev);
                        longPressHandler.removeCallbacks(longPressRunnable);
                        break;
                    }
                    case MotionEvent.ACTION_MOVE: {
                        maxPointers = Math.max(maxPointers, ev.getPointerCount());
                        float moveX = ev.getX() - tapDownX;
                        float moveY = ev.getY() - tapDownY;
                        if (moveX * moveX + moveY * moveY > touchSlopPx * touchSlopPx) {
                            longPressHandler.removeCallbacks(longPressRunnable);
                        }
                        break;
                    }
                    case MotionEvent.ACTION_POINTER_UP: {
                        longPressHandler.removeCallbacks(longPressRunnable);
                        // if a two-finger pointer lifts, we'll finalize at ACTION_UP
                        break;
                    }
                    case MotionEvent.ACTION_UP: {
                        longPressHandler.removeCallbacks(longPressRunnable);

                        boolean isSinglePointer = (maxPointers <= 1);
                        float dx = ev.getX() - tapDownX;
                        float dy = ev.getY() - tapDownY;
                        boolean tinyMove = (dx*dx + dy*dy) <= (touchSlopPx * touchSlopPx);
                        boolean shortPress = (SystemClock.uptimeMillis() - tapDownTs) < ViewConfiguration.getDoubleTapTimeout();

                        // A true long press is when the gesture was held long enough AND there was no significant movement.
                        if (wasLongPressed && tinyMove) {
                            logJson("long_press_end", true, kv("tap_id", currentTapId));
                            Scene scene = arFragment.getArSceneView().getScene();
                            HitTestResult hr = scene.hitTest(ev);
                            if (hr.getNode() instanceof TransformableNode) {
                                final Node longPressedNode = hr.getNode();
                                runOnUiThread(() -> {
                                    final CharSequence[] items = {"Delete", "Copy"};
                                    new AlertDialog.Builder(HelloSceneformActivity.this)
                                        .setTitle("Node Options")
                                        .setItems(items, (dialog, which) -> {
                                            switch (which) {
                                                case 0: // Delete
                                                    if (longPressedNode.getParent() != null) {
                                                        longPressedNode.getParent().removeChild(longPressedNode);
                                                    }
                                                    Toast.makeText(HelloSceneformActivity.this, "Deleted!", Toast.LENGTH_SHORT).show();
                                                    break;
                                                case 1: // Copy
                                                    Toast.makeText(HelloSceneformActivity.this, "Copied!", Toast.LENGTH_SHORT).show();
                                                    break;
                                            }
                                        })
                                        .show();
                                });
                            }
                            break;
                        }

                        if (isSinglePointer && tinyMove && shortPress) {
                            tapCount++;
                            Handler handler = new Handler();
                            handler.postDelayed(() -> {
                                if (tapCount == 1) {
                                    // --- Light tap: always emit one `tap` event ---
                                    String target = "empty";
                                    if (currentTapId == lastPlaceTapId) {
                                        target = "plane";
                                    } else {
                                        // hit test against nodes
                                        Scene scene = arFragment.getArSceneView().getScene();
                                        HitTestResult hr = scene.hitTest(ev);
                                        if (hr.getNode() instanceof TransformableNode) target = "node";
                                    }
                                    boolean selectedAfter = (arFragment.getTransformationSystem().getSelectedNode() != null);
                                    logJson("tap", true, kv("tap_id", currentTapId), kv("target", target), kv("selected", selectedAfter));
                                } else if (tapCount == 2) {
                                    // --- Double tap ---
                                    logJson("double_tap", true, kv("tap_id", currentTapId));
                                    Scene scene = arFragment.getArSceneView().getScene();
                                    HitTestResult hr = scene.hitTest(ev);
                                    if (hr.getNode() instanceof TransformableNode) {
                                        Node nodeToRemove = hr.getNode();
                                        nodeToRemove.getParent().removeChild(nodeToRemove);
                                    }
                                }
                                tapCount = 0;
                            }, ViewConfiguration.getDoubleTapTimeout());
                            break; // do not enter drag/pinch/rotate for taps
                        }

                        // --- Non-tap: log drag/pinch/rotate ---
                        if (downSelectedNode != null) {
                            if (maxPointers <= 1 && downLocalPos != null && downOnSelectedNode) {
                                Vector3 pos1 = downSelectedNode.getLocalPosition();
                                float dTrans = Vector3.subtract(pos1, downLocalPos).length();
                                boolean dragOk = dTrans >= EPS_T_M;
                                logJson("drag", dragOk, kv("dTrans_m", dTrans));
                            } else if (maxPointers >= 2) {
                                // Calibrate against local transform deltas if available
                                float scaleDeltaAbs = Math.abs(twoF.accumScale - 1f);
                                float rotDeltaAbs = Math.abs(twoF.accumRotateDeg);
                                if (downLocalScale != null) {
                                    Vector3 s1 = downSelectedNode.getLocalScale();
                                    float s0 = downLocalScale.x, sx1 = s1.x;
                                    float dS = Math.abs((sx1 - s0) / Math.max(1e-6f, s0));
                                    if (dS > scaleDeltaAbs) { scaleDeltaAbs = dS; twoF.accumScale = 1f + dS; }
                                }
                                if (downLocalRot != null) {
                                    float yaw0 = yawDeg(downLocalRot);
                                    float yaw1 = yawDeg(downSelectedNode.getLocalRotation());
                                    float dYawLocal = Math.abs(normalizeDeg(yaw1 - yaw0));
                                    if (dYawLocal > rotDeltaAbs) { rotDeltaAbs = dYawLocal; twoF.accumRotateDeg = (twoF.accumRotateDeg>=0? dYawLocal : -dYawLocal); }
                                }

                                // Dominance rule
                                boolean pinchPass = scaleDeltaAbs >= EPS_S;
                                boolean rotatePass = rotDeltaAbs >= EPS_R_DEG;
                                if (pinchPass && !rotatePass) {
                                    logJson("pinch", true,
                                            kv("scale_factor", twoF.accumScale),
                                            kv("dScale_abs", scaleDeltaAbs));
                                } else if (rotatePass && !pinchPass) {
                                    logJson("rotate", true,
                                            kv("dYaw_deg", twoF.accumRotateDeg));
                                } else if (pinchPass && rotatePass) {
                                    float pinchScore = scaleDeltaAbs / Math.max(1e-6f, EPS_S);
                                    float rotScore   = rotDeltaAbs / Math.max(1e-6f, EPS_R_DEG);
                                    float R = 1.6f; // dominance ratio
                                    if (pinchScore > rotScore * R) {
                                        logJson("pinch", true,
                                                kv("scale_factor", twoF.accumScale),
                                                kv("dScale_abs", scaleDeltaAbs));
                                    } else if (rotScore > pinchScore * R) {
                                        logJson("rotate", true,
                                                kv("dYaw_deg", twoF.accumRotateDeg));
                                    } else {
                                        // tie: pick the larger absolute normalized change
                                        if (pinchScore >= rotScore) {
                                            logJson("pinch", true,
                                                    kv("scale_factor", twoF.accumScale),
                                                    kv("dScale_abs", scaleDeltaAbs));
                                        } else {
                                            logJson("rotate", true,
                                                    kv("dYaw_deg", twoF.accumRotateDeg));
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            } catch (Throwable t) {
                Log.e(OP_TAG, "gesture_runtime_error", t);
            }
        });
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

    private static String vec3(Pose pose) {
        float[] t = pose.getTranslation();
        return t[0] + "," + t[1] + "," + t[2];
    }

    // -------------------- Math helpers --------------------
    private static float yawDeg(Quaternion q) {
        double t3 = 2.0 * (q.w * q.y + q.z * q.x);
        double t4 = 1.0 - 2.0 * (q.y * q.y + q.x * q.x);
        return (float) Math.toDegrees(Math.atan2(t3, t4));
    }
    private static float normalizeDeg(float a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

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
