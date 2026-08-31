/**
 * mono_video.cc — Custom monocular ORB-SLAM3 runner for the CUT3R-comparison
 * pipeline. Modelled on Examples/Monocular/mono_tum.cc.
 *
 * Usage:
 *   mono_video <vocabulary> <settings.yaml> <image_folder> <timestamps.txt> <output_dir> [--viewer]
 *
 * Emits into <output_dir>:
 *   KeyFrameTrajectory.txt  TUM format, keyframes only (via ORB-SLAM3's own saver)
 *   CameraTrajectory.txt    TUM format, every frame where tracking was OK
 *   map_points.ply          Accumulated map points, world frame, ASCII PLY
 *   tracking_log.csv        Per-frame diagnostics (THE headline output)
 *   run_meta.json           Runtime, counts, tracking-loss event summary
 *
 * ── POSE CONVENTION (important, verified by construction) ─────────────────
 * System::TrackMonocular() returns Sophus::SE3f Tcw — WORLD-TO-CAMERA.
 * TUM trajectory format expects Twc (camera-to-world): translation = camera
 * centre in world coords, quaternion = camera->world rotation.
 * We therefore invert explicitly below (`Twc = Tcw.inverse()`) before writing.
 *
 * This matters because ORB-SLAM3's System::SaveTrajectoryTUM() REFUSES to run
 * in monocular mode (it errors out — mono has no per-frame pose guarantee), so
 * CameraTrajectory.txt cannot come from the library and must be built here.
 * Since we write it ourselves, the convention is ours to fix: what lands in
 * CameraTrajectory.txt IS camera-to-world, which is what convert_trajectory.py
 * assumes. Do not "helpfully" remove the .inverse() call.
 *
 * ── MAP POINTS ────────────────────────────────────────────────────────────
 * Stock ORB-SLAM3 does not expose the Atlas publicly, so rather than patching
 * the library we accumulate the union of System::GetTrackedMapPoints() across
 * all frames into a set keyed by MapPoint*. This yields every point that was
 * ever matched during tracking — a very close proxy for the atlas contents,
 * with no library modification required. Bad/culled points are filtered at
 * write time.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>
#include <MapPoint.h>

using namespace std;

// ── Tracking state enum -> readable name ────────────────────────────────────
// Mirrors ORB_SLAM3::Tracking::eTrackingState. Logged as strings so the CSV is
// human-readable and analyze_run_orb.py can shade bands by name.
static string StateName(int s) {
    switch (s) {
        case -1: return "SYSTEM_NOT_READY";
        case  0: return "NO_IMAGES_YET";
        case  1: return "NOT_INITIALIZED";
        case  2: return "OK";
        case  3: return "RECENTLY_LOST";
        case  4: return "LOST";
        case  5: return "OK_KLT";
        default: return "UNKNOWN_" + to_string(s);
    }
}

static bool IsOkState(int s) { return s == 2 || s == 5; }

// ── Load timestamps.txt (one float per line, seconds) ───────────────────────
static bool LoadTimestamps(const string &path, vector<double> &out) {
    ifstream f(path.c_str());
    if (!f.is_open()) return false;
    string line;
    while (getline(f, line)) {
        // tolerate blank lines / trailing whitespace
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == string::npos) continue;
        out.push_back(atof(line.c_str() + first));
    }
    return true;
}

// ── Load frame_list.txt if present, else synthesise NNNNNN.<ext> names ──────
static bool LoadFrameList(const string &folder, size_t n_expected,
                          vector<string> &out) {
    ifstream f((folder + "/frame_list.txt").c_str());
    if (f.is_open()) {
        string line;
        while (getline(f, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == string::npos) continue;
            size_t last = line.find_last_not_of(" \t\r\n");
            out.push_back(line.substr(first, last - first + 1));
        }
        if (!out.empty()) return true;
    }
    // Fallback: probe for the extension extract_frames.py used.
    const char *exts[] = {"png", "jpg", "jpeg"};
    for (const char *ext : exts) {
        ostringstream probe;
        probe << folder << "/" << setw(6) << setfill('0') << 0 << "." << ext;
        ifstream test(probe.str().c_str());
        if (test.good()) {
            for (size_t i = 0; i < n_expected; ++i) {
                ostringstream name;
                name << setw(6) << setfill('0') << i << "." << ext;
                out.push_back(name.str());
            }
            return true;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        cerr << "Usage: mono_video <vocabulary> <settings.yaml> <image_folder> "
                "<timestamps.txt> <output_dir> [--viewer]" << endl;
        return 1;
    }

    const string vocab_path    = argv[1];
    const string settings_path = argv[2];
    const string image_folder  = argv[3];
    const string ts_path       = argv[4];
    const string out_dir       = argv[5];

    bool use_viewer = false;
    for (int i = 6; i < argc; ++i) {
        if (string(argv[i]) == "--viewer") use_viewer = true;
    }

    // ── Load inputs ─────────────────────────────────────────────────────────
    vector<double> timestamps;
    if (!LoadTimestamps(ts_path, timestamps) || timestamps.empty()) {
        cerr << "[ERROR] Could not read timestamps from " << ts_path << endl;
        return 1;
    }

    vector<string> frame_names;
    if (!LoadFrameList(image_folder, timestamps.size(), frame_names)) {
        cerr << "[ERROR] Could not determine frame filenames in " << image_folder
             << " (no frame_list.txt and no 000000.png/.jpg found)" << endl;
        return 1;
    }

    const size_t n_frames = min(timestamps.size(), frame_names.size());
    if (timestamps.size() != frame_names.size()) {
        cerr << "[WARN] timestamps (" << timestamps.size() << ") and frames ("
             << frame_names.size() << ") count mismatch; using " << n_frames << endl;
    }
    cout << "Frames to process: " << n_frames << endl;

    // ── Init SLAM ───────────────────────────────────────────────────────────
    cout << "Initialising ORB-SLAM3 (monocular)..." << endl;
    ORB_SLAM3::System SLAM(vocab_path, settings_path,
                           ORB_SLAM3::System::MONOCULAR, use_viewer);

    // ── Per-frame accumulators ──────────────────────────────────────────────
    struct FrameRecord {
        int    index;
        double timestamp;
        int    state;
        int    n_keypoints;
        int    n_map_matches;
        bool   is_keyframe;
        int    n_map_points_total;
        bool   pose_valid;
        double tx, ty, tz, qx, qy, qz, qw;   // Twc (camera-to-world)
    };
    vector<FrameRecord> records;
    records.reserve(n_frames);

    set<ORB_SLAM3::MapPoint *> all_map_points;

    int prev_state = -1;
    int n_lost_events = 0;          // transitions OK -> LOST/RECENTLY_LOST
    int n_reloc_events = 0;         // transitions LOST/RECENTLY_LOST -> OK
    int n_tracked_frames = 0;

    auto t_start = chrono::steady_clock::now();

    for (size_t i = 0; i < n_frames; ++i) {
        const string img_path = image_folder + "/" + frame_names[i];
        cv::Mat im = cv::imread(img_path, cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            cerr << "[WARN] Could not read image: " << img_path << " — skipping" << endl;
            continue;
        }

        const double tframe = timestamps[i];

        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, tframe);

        const int state = SLAM.GetTrackingState();
        vector<ORB_SLAM3::MapPoint *> tracked = SLAM.GetTrackedMapPoints();
        vector<cv::KeyPoint> kps = SLAM.GetTrackedKeyPointsUn();

        int n_matches = 0;
        for (auto *pMP : tracked) {
            if (pMP && !pMP->isBad()) {
                ++n_matches;
                all_map_points.insert(pMP);
            }
        }

        // State-transition bookkeeping
        if (IsOkState(prev_state) && (state == 3 || state == 4)) ++n_lost_events;
        if ((prev_state == 3 || prev_state == 4) && IsOkState(state)) ++n_reloc_events;
        prev_state = state;

        FrameRecord r;
        r.index              = static_cast<int>(i);
        r.timestamp          = tframe;
        r.state              = state;
        r.n_keypoints        = static_cast<int>(kps.size());
        r.n_map_matches      = n_matches;
        r.is_keyframe        = false;   // filled in after the run from keyframe timestamps
        r.n_map_points_total = static_cast<int>(all_map_points.size());
        r.pose_valid         = false;
        r.tx = r.ty = r.tz = r.qx = r.qy = r.qz = 0.0;
        r.qw = 1.0;

        if (IsOkState(state)) {
            // INVERT: TrackMonocular gives Tcw (world->camera); TUM wants Twc.
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f t = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            if (std::isfinite(t.x()) && std::isfinite(t.y()) && std::isfinite(t.z())) {
                r.pose_valid = true;
                r.tx = t.x(); r.ty = t.y(); r.tz = t.z();
                r.qx = q.x(); r.qy = q.y(); r.qz = q.z(); r.qw = q.w();
                ++n_tracked_frames;
            }
        }

        records.push_back(r);

        if ((i + 1) % 50 == 0 || i + 1 == n_frames) {
            cout << "  frame " << (i + 1) << "/" << n_frames
                 << "  state=" << StateName(state)
                 << "  kp=" << r.n_keypoints
                 << "  matches=" << n_matches
                 << "  map=" << r.n_map_points_total << endl;
        }
    }

    auto t_end = chrono::steady_clock::now();
    const double wall_seconds =
        chrono::duration_cast<chrono::duration<double>>(t_end - t_start).count();

    cout << "\nTracking finished in " << wall_seconds << " s. Shutting down..." << endl;
    SLAM.Shutdown();

    // ── KeyFrameTrajectory.txt (library saver — valid for monocular) ────────
    const string kf_path = out_dir + "/KeyFrameTrajectory.txt";
    SLAM.SaveKeyFrameTrajectoryTUM(kf_path);
    cout << "  Wrote " << kf_path << endl;

    // Mark which frames were keyframes, by matching timestamps against the
    // keyframe file we just wrote (cheap, avoids needing Atlas access).
    {
        ifstream kf(kf_path.c_str());
        vector<double> kf_ts;
        string line;
        while (getline(kf, line)) {
            if (line.empty() || line[0] == '#') continue;
            istringstream ss(line);
            double t;
            if (ss >> t) kf_ts.push_back(t);
        }
        sort(kf_ts.begin(), kf_ts.end());
        for (auto &r : records) {
            for (double t : kf_ts) {
                if (fabs(t - r.timestamp) < 1e-6) { r.is_keyframe = true; break; }
            }
        }
        cout << "  Keyframes identified: " << kf_ts.size() << endl;
    }

    // ── CameraTrajectory.txt (built here — see POSE CONVENTION note above) ──
    {
        const string path = out_dir + "/CameraTrajectory.txt";
        ofstream f(path.c_str());
        f << fixed << setprecision(9);
        f << "# ORB-SLAM3 monocular per-frame trajectory, TUM format\n";
        f << "# timestamp tx ty tz qx qy qz qw   (Twc: camera-to-world)\n";
        int written = 0;
        for (const auto &r : records) {
            if (!r.pose_valid) continue;
            f << r.timestamp << " "
              << r.tx << " " << r.ty << " " << r.tz << " "
              << r.qx << " " << r.qy << " " << r.qz << " " << r.qw << "\n";
            ++written;
        }
        f.close();
        cout << "  Wrote " << path << " (" << written << " poses)" << endl;
    }

    // ── map_points.ply ──────────────────────────────────────────────────────
    {
        vector<Eigen::Vector3f> pts;
        pts.reserve(all_map_points.size());
        for (auto *pMP : all_map_points) {
            if (!pMP || pMP->isBad()) continue;
            Eigen::Vector3f p = pMP->GetWorldPos();
            if (std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()))
                pts.push_back(p);
        }
        const string path = out_dir + "/map_points.ply";
        ofstream f(path.c_str());
        f << "ply\nformat ascii 1.0\n";
        f << "element vertex " << pts.size() << "\n";
        f << "property float x\nproperty float y\nproperty float z\n";
        f << "end_header\n";
        f << fixed << setprecision(6);
        for (const auto &p : pts) f << p.x() << " " << p.y() << " " << p.z() << "\n";
        f.close();
        cout << "  Wrote " << path << " (" << pts.size() << " points)" << endl;
    }

    // ── tracking_log.csv — THE headline diagnostic ──────────────────────────
    {
        const string path = out_dir + "/tracking_log.csv";
        ofstream f(path.c_str());
        f << "frame_index,timestamp,tracking_state,n_keypoints,n_map_matches,"
             "is_keyframe,n_map_points_total\n";
        f << fixed << setprecision(9);
        for (const auto &r : records) {
            f << r.index << "," << r.timestamp << "," << StateName(r.state) << ","
              << r.n_keypoints << "," << r.n_map_matches << ","
              << (r.is_keyframe ? 1 : 0) << "," << r.n_map_points_total << "\n";
        }
        f.close();
        cout << "  Wrote " << path << " (" << records.size() << " rows)" << endl;
    }

    // ── run_meta.json ───────────────────────────────────────────────────────
    {
        int n_keyframes = 0;
        for (const auto &r : records) if (r.is_keyframe) ++n_keyframes;

        // Longest contiguous non-OK run, in frames
        int longest_gap = 0, cur = 0;
        for (const auto &r : records) {
            if (!IsOkState(r.state)) { ++cur; longest_gap = max(longest_gap, cur); }
            else cur = 0;
        }

        const string path = out_dir + "/run_meta.json";
        ofstream f(path.c_str());
        f << "{\n";
        f << "  \"wall_clock_seconds\": " << fixed << setprecision(3) << wall_seconds << ",\n";
        f << "  \"frames_processed\": " << records.size() << ",\n";
        f << "  \"frames_tracked_ok\": " << n_tracked_frames << ",\n";
        f << "  \"n_keyframes\": " << n_keyframes << ",\n";
        f << "  \"n_map_points\": " << all_map_points.size() << ",\n";
        f << "  \"n_tracking_loss_events\": " << n_lost_events << ",\n";
        f << "  \"n_relocalization_events\": " << n_reloc_events << ",\n";
        f << "  \"longest_non_ok_gap_frames\": " << longest_gap << ",\n";
        f << "  \"initialized\": " << (n_tracked_frames > 0 ? "true" : "false") << ",\n";
        f << "  \"pose_convention\": \"CameraTrajectory.txt stores Twc "
             "(camera-to-world); TrackMonocular returns Tcw and is inverted here\",\n";
        f << "  \"map_point_source\": \"union of GetTrackedMapPoints() over all "
             "frames (stock ORB-SLAM3 exposes no public Atlas accessor)\"\n";
        f << "}\n";
        f.close();
        cout << "  Wrote " << path << endl;

        if (n_tracked_frames == 0) {
            cerr << "\n[ERROR] ORB-SLAM3 never initialised — zero frames tracked.\n"
                 << "  Common causes:\n"
                 << "    * Opening motion is pure rotation (mono needs TRANSLATION "
                    "for an initial baseline)\n"
                 << "    * Calibration far from reality (check settings.yaml fx/fy)\n"
                 << "    * Too few features — try raising ORBextractor.nFeatures or "
                    "lowering iniThFAST\n"
                 << "    * Frame rate too low / motion too fast between frames\n";
            return 2;
        }
    }

    cout << "\nDone. Outputs in " << out_dir << endl;
    return 0;
}
