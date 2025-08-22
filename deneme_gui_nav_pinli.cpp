// 	g++ -std=c++17 -Wall -o deneme deneme_gui_nav.cpp `pkg-config --cflags --libs opencv4` 
// ./darknet detector demo ./lyec13/lyec13.data ./lyec13/lyec13.cfg ./lyec13/lyec13b.weights -c 2 -json_port 8888 -thresh 0.25 //pinli en son

//g++ -std=c++17 -o example example.cpp `pkg-config --cflags --libs opencv4` -I/path/to/darkhelp/include -L/path/to/darkhelp/lib -ldarkhelp

//derlemek : g++ -std=c++17 -Wall -o pinli deneme_gui_nav_pinli.cpp `pkg-config --cflags --libs opencv4` -I/path/to/darkhelp/include -L/path/to/darkhelp/lib -ldarkhelp -ljetgpio


#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
//#include <boost/asio.hpp>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <list>
#include <vector>
#include <functional>
#include <memory>
#include <iterator>
#include <sstream>
#include <unistd.h>
#include <numeric>
#include <jetgpio.h>
#include <chrono>
#include <DarkHelp.hpp>

using namespace std::chrono;
using namespace cv;
using namespace std;
using namespace std::chrono_literals;

    // Camera setup
VideoCapture cap_serit(2); 
VideoCapture cap_arka(0);

double fps = cap_arka.get(CAP_PROP_FPS);

/*
Point blackTriangle[1][3];
blackTriangle[0][0] = Point(20, 70);
blackTriangle[0][1] = Point(70, 70);
blackTriangle[0][2] = Point(45, 20);

const Point* pts1[1] = {blackTriangle[0]};
int npts[] = {3};

fillPoly(ikaz_img_1, pts1, npts, 1, Scalar(0,0,0));

Point yellowTriangle[1][3];
yellowTriangle[0][0] = Point(30, 60);
yellowTriangle[0][1] = Point(60, 60);
yellowTriangle[0][2] = Point(45, 30);

const Point* pts2[1] = {yellowTriangle[0]};
int npts2[] = {3};

fillPoly(ikaz_img_1, pts2, npts2, 1, Scalar(0,255,255));
*/


// GPIO pins
const int led = 8;
const int anahtar = 10;
bool arkaCamOpen = false;
bool seritUyari = false; // led 0 ken true 1 ken false olcak (led default da yanıyor da )
struct MapPoint {
    string name;
    double lat;
    double lon;
    int x; 
    int y;
};

int roi_top_y = 400;   // üst çizgi yüksekliği
int roi_offset = 100;  // ortadaki boşluk
int roi_left_x = 50;   // sol alt x
int roi_right_x = 50;  // sağ alt x

int offset= -999; //bu fark / led ynması buna göre

vector<Point> getROI(int width, int height) {
    vector<Point> vertices;
    vertices.push_back(Point(roi_left_x, height));                // sol alt
    vertices.push_back(Point(width/2 - roi_offset, roi_top_y));   // üst sol
    vertices.push_back(Point(width/2 + roi_offset, roi_top_y));   // üst sağ
    vertices.push_back(Point(width - roi_right_x, height));       // sağ alt
    return vertices;
}


const int FILTER_SIZE = 5;
list<double> latBuffer;
list<double> lonBuffer;
int class_id = -1;

    string previousNearest = "";
    set<string> visitedPoints;
    int lap = 0; //aracın tur saısı

//konumlar
vector<MapPoint> mapPoints = {
        {"start", 41.4545428343641, 31.7673288210482, 130,50},
        {"1.",    41.4545683153017, 31.7673578926497 ,149,52},
        {"2.",    41.4545726130563, 31.767403006056, 149,52},
        {"3.",    41.4545769466599, 31.7674423448921, 184,51},
        {"4.",    41.4545824840915, 31.767490028757, 227,52},
        {"5.",    41.4545858704149, 31.7675282791321,287,55},
        {"6.",    41.4545975855043, 31.7675809654595, 333,54},
        {"7.",    41.4546046916607, 31.7676301432937, 385,51},
        {"8.",    41.4546140570198, 31.7676957467626, 442,50},
        {"9.",    41.4546138553442, 31.7677417011851, 496,53},
        {"10.",   41.4545716450857, 31.767733329002, 510, 171},
        {"11.",   41.4545344344884, 31.7677323558517, 554,44},
        {"12.",   41.454492751459,  31.7677326504592, 571,80},
        {"13.",   41.4544663150371, 31.7676935868685, 586,152},
        {"14.",   41.4544669420935, 31.767653177944, 535,146},
        {"15.",   41.4544640682458, 31.7676120910883, 434,150},
        {"16.",   41.454468593804,  31.7675666947408, 396,179},
        {"17.",   41.4544700454515, 31.7675200978711, 348,216},
        {"18.",   41.4544728933769, 31.7674694353018, 284,213},
        {"19.",   41.4544758090147, 31.7674212671161, 250,170},
        {"20.",   41.4544758285762, 31.7673628247006, 241,119},
        {"21.",   41.4544841396325, 31.7673226915122, 220,82},
        {"22.",   41.4545167638021, 31.7673173271021, 180,59}
    };

const double R = 6371.0;

// haversine ---------------------------------------------------------------------------
double haversine(double lat1, double lon1, double lat2, double lon2) {
    double dLat = (lat2 - lat1) * M_PI / 180.0;
    double dLon = (lon2 - lon1) * M_PI / 180.0;
    lat1 = lat1 * M_PI / 180.0;
    lat2 = lat2 * M_PI / 180.0;
    double a = sin(dLat/2)*sin(dLat/2) + sin(dLon/2)*sin(dLon/2)*cos(lat1)*cos(lat2);
    double c = 2*atan2(sqrt(a), sqrt(1-a));
    return R*c;
}

//nmea2decimal --------------------------------------------------------------------------
double convertToDecimalDegrees(const string& nmea, const string& direction) {
    if (nmea.empty()) return 0.0;
    try {
        double raw = stod(nmea);
        int degrees = static_cast<int>(raw / 100);
        double minutes = raw - (degrees * 100);
        double decimal = degrees + minutes / 60.0;
        if (direction == "S" || direction == "W") decimal *= -1;
        return decimal;
    } catch (...) {
        return 0.0;
    }
}
//konumbulanzi-----------------------------------------------------------------------------
cv::Point konumxy() {
    ifstream serial("/dev/ttyACM0");
    if (!serial.is_open()) {
        cerr << "Seri port açılamadı" << endl;
    }
    
    
    
    string line;
    while (getline(serial, line)) {
        size_t pos = line.find("$GNGLL");
        if (pos != string::npos) {
            string gll = line.substr(pos);
            stringstream ss(gll);
            string token;
            vector<string> fields;
            while (getline(ss, token, ',')) fields.push_back(token);

            if (fields.size() >= 5 && !fields[1].empty() && !fields[3].empty()) {
                double currentLat = convertToDecimalDegrees(fields[1], fields[2]);
                double currentLon = convertToDecimalDegrees(fields[3], fields[4]);
                
            if (currentLat == 0.0 || currentLon == 0.0) {
        		cerr << "Geçersiz GPS verisi (0,0) alındı, atlanıyor." << endl;
        		continue; // bu satırı işleme alma, yeni veri bekle
    		}
                
                // ------------------ MOVING AVERAGE FILTRE ------------------
                latBuffer.push_back(currentLat);
                lonBuffer.push_back(currentLon);
                if(latBuffer.size() > FILTER_SIZE) {
                    latBuffer.pop_front();
                    lonBuffer.pop_front();
                }
                double avgLat = accumulate(latBuffer.begin(), latBuffer.end(), 0.0) /latBuffer.size();
                double avgLon = accumulate(lonBuffer.begin(), lonBuffer.end(), 0.0) / lonBuffer.size();
                currentLat = avgLat;
                currentLon = avgLon;
                //

                double minDist = 1e9;
                MapPoint nearestPoint;
                for (auto &p : mapPoints) {
                    double d = haversine(currentLat, currentLon, p.lat, p.lon);
                    if (d < minDist) {
                        minDist = d;
                        nearestPoint = p;
                    }
                }

                if (nearestPoint.name != "start") {
                    visitedPoints.insert(nearestPoint.name);
                    cout << "visitedpoint"<< visitedPoints.size() << endl;
                }

                if (nearestPoint.name == "start" && previousNearest != "start") {
                   // if (visitedPoints.size() == mapPoints.size() - 1) {
                     //   lap++;
                   // }
                   if (visitedPoints.size() >= mapPoints.size() - 4) {
                        lap++;
                    }
                    visitedPoints.clear();
                }
                previousNearest = nearestPoint.name;

                cout << fixed << setprecision(6);
                cout << "Anlik konum: Lat=" << currentLat
                     << " Lon=" << currentLon
                     << " | En yakin nokta: " << nearestPoint.name
                     << " (Mesafe: " << minDist << " km)"
                     << " | Tur: " << lap << endl;

                return Point(nearestPoint.x, nearestPoint.y);
            }
        }
    }

    // Eğer hiç değer bulunmazsa varsayılan 0,0 
    return Point(0, 0);
}

//yapay zeka ---------------------------------------------------------------------------------------
cv::Mat predictAndAnnotate(DarkHelp::NN &nn, const cv::Mat &frame)
{
    const auto results = nn.predict(frame);

    std::cout << results << std::endl;
    if(results.empty())
    {
    	class_id = -1;
    }
    else{
    	for(const auto &det : results)
    {
    	class_id = det.best_class;
    	std::cout << "best class: " << class_id << std::endl;
    }
    }
    
    return nn.annotate();
}
//--------------------------------------------------------------------------------------

Mat grayscale(const Mat &img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}

Mat gaussian_blur(const Mat &img, int kernel_size) {
    Mat blur;
    GaussianBlur(img, blur, Size(kernel_size, kernel_size), 0);
    return blur;
}

Mat canny(const Mat &img, double low_threshold, double high_threshold) {
    Mat edges;
    Canny(img, edges, low_threshold, high_threshold);
    return edges;
}

Mat region_of_interest(const Mat &img, const vector<Point> &vertices) {
    Mat mask = Mat::zeros(img.size(), img.type());
    if (img.channels() > 1)
        fillPoly(mask, vector<vector<Point>>{vertices}, Scalar(255,255,255));
    else
        fillPoly(mask, vector<vector<Point>>{vertices}, Scalar(255));
    Mat masked;
    bitwise_and(img, mask, masked);
    return masked;
}

Mat weighted_img(const Mat &img, const Mat &initial_img, double alpha=0.8, double beta=1.0, double gamma=0.0) {
    Mat result;
    addWeighted(initial_img, alpha, img, beta, gamma, result);
    return result;
}

// Segmentleri birleştirip tek çizgiye çevir
void average_slope_intercept(const vector<Vec4i> &lines, int img_rows, int img_cols,
                             Vec4i &left_line, Vec4i &right_line) {
    vector<double> left_slopes, left_intercepts;
    vector<double> right_slopes, right_intercepts;

    for(auto l : lines){
        double x1 = l[0], y1 = l[1];
        double x2 = l[2], y2 = l[3];
        if(x2 - x1 == 0) continue; // Sonsuz eğim

        double slope = (y2 - y1) / (x2 - x1);
        double intercept = y1 - slope * x1;

        if(slope < -0.3){
            left_slopes.push_back(slope);
            left_intercepts.push_back(intercept);
        }
        else if(slope > 0.3){
            right_slopes.push_back(slope);
            right_intercepts.push_back(intercept);
        }
    }

    auto mean = [](const vector<double> &v) {
        if(v.empty()) return 0.0;
        double sum = 0;
        for(double x : v) sum += x;
        return sum / v.size();
    };

    double left_slope = mean(left_slopes);
    double left_intercept = mean(left_intercepts);
    double right_slope = mean(right_slopes);
    double right_intercept = mean(right_intercepts);

    int y1 = img_rows;
    int y2 = int(img_rows * 0.6);

    if(left_slopes.size() > 0){
        int x1 = int((y1 - left_intercept)/left_slope);
        int x2 = int((y2 - left_intercept)/left_slope);
        left_line = Vec4i(x1, y1, x2, y2);
    }

    if(right_slopes.size() > 0){
        int x1 = int((y1 - right_intercept)/right_slope);
        int x2 = int((y2 - right_intercept)/right_slope);
        right_line = Vec4i(x1, y1, x2, y2);
    }
}

// Hough çizgilerini çiz (parça parça)
void draw_lines(Mat &img, const vector<Vec4i> &lines, const Scalar &color=Scalar(0,0,255), int thickness=2) {
    for(const auto &l : lines){
        line(img, Point(l[0], l[1]), Point(l[2], l[3]), color, thickness);
    }
}

// Hough ve average çizgileri hesapla
void hough_lines(const Mat &img, Mat &segments_img, Mat &average_img) {
    vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, CV_PI/180, 5, 20, 10);

    // Parça parça çizgiler
    segments_img = Mat::zeros(img.size(), CV_8UC3);
    draw_lines(segments_img, lines);

    // Average çizgiler
    Vec4i left_line(0,0,0,0), right_line(0,0,0,0);
    average_slope_intercept(lines, img.rows, img.cols, left_line, right_line);
    average_img = Mat::zeros(img.size(), CV_8UC3);
    if(left_line != Vec4i(0,0,0,0) || right_line != Vec4i(0,0,0,0))
        draw_lines(average_img, {left_line, right_line}, Scalar(0,255,0), 5);

    // Şerit ortası - frame ortası farkını yaz
    if(left_line != Vec4i(0,0,0,0) && right_line != Vec4i(0,0,0,0)) {
        int lane_center = (left_line[0] + right_line[0]) / 2;
        int frame_center = img.cols / 2;
        offset = lane_center - frame_center;

        string text = "Offset: " + to_string(offset);
        putText(average_img, text, Point(50,50), FONT_HERSHEY_SIMPLEX, 
                1, Scalar(255,255,255), 2);
        line(average_img, Point(frame_center, img.rows), Point(frame_center, img.rows-100), Scalar(255,0,0), 2); // frame ortası
        line(average_img, Point(lane_center, img.rows), Point(lane_center, img.rows-100), Scalar(0,0,255), 2);   // şerit ortası
    }
}


Mat Lanexx(Mat frame, int thresh){

Mat gray = grayscale(frame);
        Mat blur_gray = gaussian_blur(gray, 5);
        Mat edges = canny(blur_gray, 50, 150);

        // ROI
        vector<Point> vertices = getROI(frame.cols, frame.rows);
        Mat masked_edges = region_of_interest(edges, vertices);

        // Hough + Average
        Mat segments_img, average_img;
        hough_lines(masked_edges, segments_img, average_img);

        // Orijinal üzerine çizgiler bindir
        Mat result = weighted_img(average_img, frame);
	
	if(offset >= thresh || offset <= (-1*thresh) )
	{//led yak uyarı ver
	//cout << " serit uyarı TRUE" << endl;
		seritUyari = true;
	}
	else
	{
	//cout << " serit uyarı FALSE" << endl;
		seritUyari = false;
	}
	
	// ROI görselleştirme (şeffaf sarı)
        Mat overlay = result.clone();
        fillPoly(overlay, vector<vector<Point>>{vertices}, Scalar(0,255,255));
        addWeighted(overlay, 0.3, result, 0.7, 0, result);

	return result;
}



//---------------------------------------------------------------------------------------
//gui

Mat GUI(int sayac,Mat serit, Mat arkaCam, Mat map, Mat tabela, cv::Point konum, string turSayisi, bool anahtar, int ID = 1, string seritDurum = "şeritte") {
    int genislik = 1280;
    int yukseklik = 720;
    
    
    Mat pencere(yukseklik, genislik, CV_8UC3, Scalar(30, 30, 30)); 
    
    putText(pencere, to_string(sayac), Point(1000, 100),
            FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 204), 2);
    
    //---------------------------PANEL--------------------------------------------
    rectangle(pencere, Point(0, 0), Point(genislik, 70), Scalar(50, 50, 50), FILLED);
    putText(pencere, "ADAS SISTEM ARAYUZU", Point(330, 55),
            FONT_HERSHEY_SIMPLEX, 1.8, Scalar(255, 255, 255), 4, LINE_AA);
            
    if(anahtar){ //arka kamera // araç geri viteste ise 
    if (!arkaCam.empty()) {
        Mat arka_fixed;
        flip(arkaCam, arka_fixed, 0);
        resize(arkaCam, arka_fixed, Size(1200, 640));
        arka_fixed.copyTo(pencere(Rect(40, 75, arka_fixed.cols, arka_fixed.rows)));
        
        putText(pencere, to_string(fps), Point(50, 70 ),
            FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 2);
    }
    }  
    else  {   
    //-----------------------------NAVIGASYON-------------------------------------
    rectangle(pencere, Point(40, 100), Point(660, 380), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Yaris Navigasyonu", Point(50, 95 ),
            FONT_HERSHEY_SIMPLEX, 0.89, Scalar(255, 255, 255), 2);
    //--------------------------------------tur sayısı //////////////--------------
    rectangle(pencere, Point(680, 100), Point(790, 140), Scalar(255, 255, 204), FILLED);
    putText(pencere, turSayisi, Point(685, 130 ),//turSayisi degiskeni int mainde "tur:" + tur
            FONT_HERSHEY_SIMPLEX, 1.1, Scalar(0, 0, 0), 2);
            
    if (!map.empty()) {
        Mat map_fixed;
        resize(map, map_fixed, Size(600, 260));
        circle(map_fixed, konum, 8, Scalar(0,0,255), FILLED);
        map_fixed.copyTo(pencere(Rect(50, 110, map_fixed.cols, map_fixed.rows)));
        
        
    } else {
        putText(pencere, "Veri Yok", Point(50, 95 ), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 200, 200), 2);
    }
    //----------------------------------------------------------------------------------------------------
    
    
    //------------------------yapayzekA----------------------------
    rectangle(pencere, Point(40, 410), Point(540, 710), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Tabela Tanima Sistemi", Point(50, 400 ),
            FONT_HERSHEY_SIMPLEX, 0.89, Scalar(255, 255, 255), 2);
    if(!tabela.empty()){
    	Mat tabela_fixed;
    	resize(tabela, tabela_fixed, Size(480, 280));
    	tabela_fixed.copyTo(pencere(Rect(50, 420, tabela_fixed.cols, tabela_fixed.rows)));
    }
    //---------tabela görsel
    
    rectangle(pencere, Point(840, 120), Point(1100, 380), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Tabela", Point(850, 100 ),
            FONT_HERSHEY_SIMPLEX, 0.89, Scalar(255, 255, 255), 2);
    Mat tabelaV;
    if(ID != -1)
    {
    	Mat tabelaV_fixed;
    	string tabelaView = "./tabelalar/" + to_string(ID) + ".png";
    	tabelaV = imread(tabelaView);
    	if(!tabelaV.empty()){    	resize(tabelaV, tabelaV_fixed, Size(240, 240));
    	tabelaV_fixed.copyTo(pencere(Rect(850, 130, tabelaV_fixed.cols, tabelaV_fixed.rows)));
    	}
    }
    else{
    	Mat imag(240, 240, CV_8UC3, Scalar(204,204,255));
    	imag.copyTo(pencere(Rect(850, 130, imag.cols, imag.rows)));
    }
    
    //serit -------------------------------------------------------------------
    rectangle(pencere, Point(560, 410), Point(1060, 710), Scalar(70, 70, 90), FILLED);
    putText(pencere, "Serit Takip Uyari Sistemi", Point(570, 400 ),
            FONT_HERSHEY_SIMPLEX, 0.89, Scalar(255, 255, 255), 2);
    if (!serit.empty()) {
        Mat serit_fixed;
        resize(serit, serit_fixed, Size(480, 280));
        serit_fixed.copyTo(pencere(Rect(570, 420, serit_fixed.cols, serit_fixed.rows)));
        
        
    } else {
        putText(pencere, "Veri Yok", Point(570, 95 ), 
                FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 200, 200), 2);
    }
    //------------------------------guccuk uyarı
    rectangle(pencere, Point(680, 160), Point(790, 270), Scalar(70, 70, 90), FILLED);
    if(seritUyari)
    {
    	Mat  ikaz_img_1(90, 90, CV_8UC3, Scalar(0, 255, 255));
	string unlem = "!";
	cv::putText(ikaz_img_1, unlem, Point(35, 60), cv::FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0,0,0), 4 );
	
	
	
    	if (!ikaz_img_1.empty()) {
        ikaz_img_1.copyTo(pencere(Rect(690, 170, ikaz_img_1.cols, ikaz_img_1.rows)));	
        }
    }
    else
    {
    Mat  ikaz_img_2(90, 90, CV_8UC3, Scalar(190, 255,0));
	cv::putText(ikaz_img_2, R"(\/)", Point(15, 55), cv::FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,0), 4 );
    	if (!ikaz_img_2.empty()) {
        ikaz_img_2.copyTo(pencere(Rect(690, 170, ikaz_img_2.cols, ikaz_img_2.rows)));	
        }
    }
}
    return pencere;
}
//-------------------------------------------------------------------------


int main(int argc, char *argv[]) {

   bool anahtarVal = false;
   int level = 0;
    // GPIO initialization--------------------------------------------------------------------------
    int Init = gpioInitialise();
    if (Init < 0) {
        cerr << "Jetgpio initialization failed. Error code: " << Init << endl;
        return Init;
    }
    cout << "Jetgpio initialized successfully." << endl;
    gpioSetMode(anahtar, JET_INPUT);

    if (!cap_arka.isOpened()) {
        cerr << "Arka Kamera acilamadi!" << endl;
    	return -1;
    }
    
    Mat map_img = imread("./map_.jpeg");
    
    if (map_img.empty()) {
       		cout << "map açılmıyor" << endl;
       		return -1;
        } 
        
    auto lastKonumUpdate = steady_clock::now();  // zaman tutucu
    Point arac_konum = Point(0, 0);
    
    Mat arka_;
    Mat serit_;
    Mat tabela_;
    int sayac = 0;
    
    //yapay zeka initialization----------------------------------------------------------------------------
    DarkHelp::Config cfg("speed_limitv4.cfg", "speed_limitv4_best.weights", "speed_limitv4.names");
        cfg.enable_tiles                    = false;
        cfg.annotation_auto_hide_labels     = false;
        cfg.annotation_include_duration     = true;
        cfg.annotation_include_timestamp    = false;
        cfg.threshold                       = 0.5f;

        DarkHelp::NN nn(cfg);
        nn.config.annotation_line_thickness = 1;
        nn.config.annotation_shade_predictions = 0.36f;
    
    namedWindow("ADAS System",  cv::WINDOW_NORMAL);
    setWindowProperty("ADAS System", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    /* cv::createTrackbar("Top Y", "ADAS System", NULL, 720);
	cv::createTrackbar("Offset", "ADAS System", NULL, 400);
	cv::createTrackbar("Left X", "ADAS System", NULL, 640);
	cv::createTrackbar("Right X", "ADAS System", NULL, 640);
*/
    
    while(true){
    	
    	level = gpioRead(anahtar);
        
        if(level == 1)
        anahtarVal = false;
        else if(level == 0)
        anahtarVal = true;
       
       if(seritUyari)
       {
	cout << " serit uyarı TRUE" << endl;
       gpioWrite(led, 1);
       }
       else
       {
       cout << " serit uyarı FALSE" << endl;
       gpioWrite(led, 0);
       }
       
    	cap_arka >> arka_;
   	cap_serit >> serit_;
    	tabela_ = serit_.clone();
        
        if (serit_.empty()) {
       		cout << "serit kamerası açılmıyor" << endl;
       		return -1;
        }
        if (tabela_.empty()) {
       		cout << "tabela kamera açılmıyor" << endl;
       		return -1;
        }
        if (arka_.empty()) {
       		cout << "arka kamera açılmıyor" << endl;
       		return -1;
        }
        auto now = steady_clock::now();
	if (duration_cast<seconds>(now - lastKonumUpdate).count() >= 1) {
		arac_konum = konumxy(); // GPS'ten konumu oku
        	lastKonumUpdate = now;  // zamanı güncelle
	}
	
	string tur_yazdir = "Tur:" + to_string(lap);
        
        //yz 
        cv::Mat annotated = predictAndAnnotate(nn, tabela_);
        Mat seritView = Lanexx(serit_, 40);
        
        // GUI oluştur
        Mat gui = GUI(sayac, seritView, arka_, map_img, annotated, arac_konum, tur_yazdir , anahtarVal, class_id);
        imshow("ADAS System", gui);
     	/*roi_top_y  = cv::getTrackbarPos("Top Y", "ADAS System");//400
	roi_offset = cv::getTrackbarPos("Offset", "ADAS System");
	roi_left_x = cv::getTrackbarPos("Left X", "ADAS System");
	roi_right_x= cv::getTrackbarPos("Right X", "ADAS System");
     */
        if(sayac > 400)
        sayac = 0;
        sayac++;
        
        int key = waitKey(30);
	if (key == 27) break; // ESC ile çık
	/*if (key == 's' || key == 'S') {
    		arkaCamOpen = !arkaCamOpen; // toggle anahtar*/
    	cout << "Anahtar durumu degisti: " << (anahtarVal ? "Açık" : "Kapalı") << endl;
}
//cap_arka.release();
    //gpioTerminate();
    destroyAllWindows();

    return 0;
    }
  
