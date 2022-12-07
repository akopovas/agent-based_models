#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <vector>   

#include "flamegpu/flamegpu.h"

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <GL/freeglut.h>
#include <GL/glut.h>

// Grid Size Размерность дискретного пространства в пикселах
#define GRID_WIDTH 800
#define GRID_HEIGHT 800
#define dim 100 // Размерность пространства 1000 x 1000
#define side_size 8 // Размер стороны квадрата
#define TIME_STOP 80 // Количество моментов времени (лет)
#define RUN_COUNT 100 // Количество прогонов для ансамбля
#define NumberOfCitizens 1000
#define NumberOfMigrants 100

// Visualisation mode (1=standalone run, 0 = essemble run)
#define VIS_MODE 1


std::mutex m;

int window_width = 1050;
int window_height = 1050;

int window_id = 0;

float x_a[100000], y_a[100000]; //Координаты агентов
int agent_type[100000]; // Тип агеннта
int a_size = 0; // Количество агентов


std::vector <double> x_cell; // Коордианты всех ячеек в непрерывном пространстве
std::vector <double> y_cell; // 

uint8_t resource_type[dim][dim]; // Тип ресурса

std::ofstream out("results.txt", std::ios::app);
std::ofstream out2("log.txt", std::ios::app); // Лог для ансамбля моделей


std::atomic<bool> occupied_cells[dim][dim] = { }; // Занятые или свободные ячейки дикретного пространства

void display(void);

extern void initVisualisation();
extern void runVisualisation();

__shared__ unsigned int agent_nextID;

__host__ __device__  unsigned int getNextID() {
    agent_nextID++;
    return agent_nextID;
}

//Сложные вычисления уровня агента
__host__ __device__ double kids_p(unsigned int kids, unsigned int current_age)
{
double w = 0.0;
if (kids > 0 && current_age > 0)
w = (double)exp(-1 / (pow(kids, 0.9) * pow(current_age, 0.1)));
return w;
}

class Cells {
public:
  //  std::mutex mtx;
    float x;
    float y;
    int r;
    int c;
    int ClusterIndex = -1; // Индекс кластера ячейки
    
Cells (float x, float y, int r, int c) {
     //std::lock_guard<std::mutex> lock(mtx);
       this->x = x;
       this->y = y;
       this->r = r;
       this->c = c;
       ClusterIndex = -1;
    }
};


// Расчёт индекса сегрегации Дункана
__host__ double Dunkan_segregation_index(int locals_count, int foreigns_count, std::vector< std::vector<int> >& agent_in_cell)
{
    double DSI = 0;
    int count_1 = 0;
    int count_2 = 0;
    int i_start = 0;
    int j_start = 0;
    int vicinity_size = 33; // Размерность окрестности = (int) dim (100) / 3

    for (int v_j = 0; v_j < vicinity_size; v_j++)
    {
        i_start = 0;
        for (int v_i = 0; v_i < vicinity_size; v_i++)
        {
            count_1 = 0;
            count_2 = 0;

            for (int j = j_start; j < j_start + 3; j++)
            {
                for (int i = i_start; i < i_start + 3; i++)
                {
                    if (agent_in_cell[i][j] == 1)
                        count_1++;
                    if (agent_in_cell[i][j] == 2)
                        count_2++;
                }
            }

            if (locals_count > 0 && foreigns_count > 0)
                DSI = DSI + 0.5 * abs((double)count_1 / locals_count - (double)count_2 / foreigns_count);

            i_start = i_start + 3;
        }
        j_start = j_start + 3;
    }

    if (locals_count == 0 || foreigns_count == 0)
        DSI = 1;

    return DSI;
}


//********************************* КЛАССЫ ДЛЯ КЛАСТЕРИЗАЦИИ АГЕНТОВ (РАБОЧИХ МЕСТ) *******************************************
class ClusterCentroid {
    // Центр кластера
private:
    double X;
    double Y;
public:
    double getClusterX() { return X; };
    double getClusterY() { return Y; };
    void setClusterX(double x) { this->X = x; };
    void setClusterY(double y) { this->Y = y; };

    ClusterCentroid(Cells p)
    {
        this->X = p.x;
        this->Y = p.y;
    }
};

// ClusterPoint
class ClusterPoint {
    // Координаты центра кластера
private:
    double X;
    double Y;
    int ClusterIndex;

    /// Gets or sets X-coord of the point
    /// 
public:
    double getX() { return X; }
    double getY() { return Y; }
    void setX(double x) { this->X = x; }
    void setY(double y) { this->Y = y; }

    ClusterPoint(Cells p)
    {
        this->X = p.x;
        this->Y = p.y;
        this->ClusterIndex = -1;
    }
};


//CMeansAlgorithm
class CMeansAlgorithm {

    /// Array containing all points-agents used by the algorithm
private:
    std::vector<Cells> Points;
    /// Array containing all clusters-agents handled by the algorithm
    std::vector<Cells> Clusters;
    std::vector<Cells> centrs;
    double Fuzzyness;
    /// Algorithm precision
    double Eps = pow(10, -5);
    double getJ() { return J; }
    void setJ(double j) { this->J = j; }


    /// Recalculates cluster indexes
    void RecalculateClusterIndexes(std::vector<Cells>& Points)
    {
        for (int i = 0; i < Points.size(); i++)
        {
            double max = -1.0;

            for (int j = 0; j < Clusters.size(); j++)
            {
                if (max < U[i][j]) // && U[i][j] >= 0.6
                {
                    max = U[i][j];
                    Points[i].ClusterIndex = j;
                }
            }
        }
    }
    /// CalculateObjectiveFunction
    double CalculateObjectiveFunction(std::vector<Cells>& Points, int max_rank, double max_distance)
    {
        double Jk = 0;

        for (int i = 0; i < Points.size(); i++)
        {
            for (int j = 0; j < Clusters.size(); j++)
            {
                Jk += pow(U[i][j], Fuzzyness) * (pow(Points[i].x - centrs[j].x, 2) + pow(Points[i].y - centrs[j].y, 2));
            }
        }
        return Jk;
    }

/// Calculates the centroids of the clusters 
    void CalculateClusterCenters(std::vector<Cells>& Points, int max_rank, double max_distance)
    {
        for (int j = 0; j < Clusters.size(); j++)
        {
            double uX = 0.0;
            double uY = 0.0;
            double l = 0.0;

            for (int i = 0; i < Points.size(); i++)
            {
                ClusterPoint p(Points[i]);
                double uu = pow(U[i][j], Fuzzyness);

                uX += uu * p.getX();
                uY += uu * p.getY();
                l += uu;
            }
            
            centrs[j].x = uX / l;
            centrs[j].y = uY / l;
        }
    }



public:
    std::vector< std::vector<double> > U;

    double J; // Minimizing criterion

    double getCentersX(int ClusterIndex)
    {
        double c_x = 0.0;
        c_x = centrs.at(ClusterIndex).x;
        return c_x;
    }

    double getCentersY(int ClusterIndex)
    {
        double c_y = 0.0;
        c_y = centrs.at(ClusterIndex).y;
        return c_y;
    }


/// 
/// Perform one step of the algorithm
/// 
    void Step(std::vector<Cells>& Points, int max_rank, double max_distance)
    {
        double diff;
        for (int i = 0; i < Points.size(); i++)
        {
            ClusterPoint p(Points[i]);
            double sum = 0.0;
            for (int j = 0; j < Clusters.size(); j++)
            {
                diff = sqrt(pow(p.getX() - centrs[j].x, 2.0) + pow(p.getY() - centrs[j].y, 2.0));
                diff = (diff == 0) ? Eps : diff;
                U[i][j] = 1.0 / pow(diff, 2.0 / (Fuzzyness - 1.0));
                sum += U[i][j];
            }

            for (int j = 0; j < Clusters.size(); j++)
            {
                U[i][j] = U[i][j] / sum;
            }
        }

        RecalculateClusterIndexes(Points);
    }

    /// 
    /// Perform a complete run of the algorithm until the desired accuracy is achieved.
    /// For demonstration issues, the maximum Iteration counter is set to 20.
    /// 
    /// Algorithm accuracy
    /// The number of steps the algorithm needed to complete
    int Run(double accuracy, std::vector<Cells>& Points, int max_rank, double max_distance)
    {
        int i = 0;
        int maxIterations = 100;
        do
        {
            i++;
            J = CalculateObjectiveFunction(Points, max_rank, max_distance);
            CalculateClusterCenters(Points, max_rank, max_distance);
            Step(Points, max_rank, max_distance);
            double Jnew = CalculateObjectiveFunction(Points, max_rank, max_distance);
            if (abs(J - Jnew) < accuracy) break;
        } while (maxIterations > i);
        return i;
    }


    CMeansAlgorithm(std::vector<Cells>& Points, std::vector<Cells>& clusters, float fuzzy) {

        this->Clusters = clusters;

         U.assign(Points.size(), std::vector<double>(this->Clusters.size()));
       // U.resize(Points.size());

        this->Fuzzyness = fuzzy;

        double diff;

        // Iterate through all points to create initial U matrix
        for (int i = 0; i < Points.size(); i++)
        {
            ClusterPoint p(Points[i]);
            double sum = 0.0;

            for (int j = 0; j < Clusters.size(); j++)
            {
                centrs.push_back(Clusters[j]);
                diff = sqrt(pow(p.getX() - centrs[j].x, 2.0) + pow(p.getY() - centrs[j].y, 2.0));
                diff = (diff == 0) ? Eps : diff;
                U[i][j] = 1.0 / pow(diff, 2.0 / (Fuzzyness - 1.0));
                sum += U[i][j];
            }

            for (int j = 0; j < this->Clusters.size(); j++)
            {
                U[i][j] = U[i][j] / sum;
            }
        }
        RecalculateClusterIndexes(Points);
    }
};
//************************************************************************************************************************
int initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
        "GL_ARB_pixel_buffer_object"
    )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return FALSE;
    }
    // default initialization
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

void reshape(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void windowResize(int width, int height) {
    window_width = width;
    window_height = height;
}

extern void initVisualisation()
{
    // Create GL context
    int   argc = 1;
    char glutString[] = "GLUT application";
    char* argv[] = { glutString, NULL };
    //char *argv[] = {"GLUT application", NULL};	

    glutInit(&argc, argv);


    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    window_id = glutCreateWindow("FLAME GPU Visualiser");
    glutReshapeFunc(windowResize);

    // initialize GL
    if (FALSE == initGL()) {
        return;
    }

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    //Здесь можно поэкспериментировать
}

extern void runVisualisation()
{
    // start rendering mainloop
    glutMainLoop();


}

void display()
{
    glutSetWindow(window_id);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    //Визуализация решетки в которой отображаются агенты

    double x = 0;
    double y = 0;
    for (int j = 0; j < dim; j++)
    {
        x = 0;
        for (int i = 0; i < dim; i++)
        {
            if (resource_type[i][j] == 0)
                glColor3ub(230, 230, 250); //lavander (нет рабочего места)
            if (resource_type[i][j] == 1)
                glColor3ub(144, 238, 144); // lightGreen (низкотехнологичное рабочее место)
            if (resource_type[i][j] == 2)
                glColor3ub(0, 128, 0); // Green (выосоктехнологичное рабочее место)

            glRectd(x, y, x + side_size, y + side_size);

            glLineWidth(1);       // ширину линии
            glBegin(GL_LINE_LOOP);
            glColor3d(0, 0, 0);     // черный цвет
            //Внешний контур
            glVertex2d(x, y);
            glVertex2d(x, y + side_size);
            glVertex2d(x + side_size, y + side_size);
            glVertex2d(x + side_size, y);
            glVertex2d(x, y);
            glEnd();

            x += side_size;
        }
        y += side_size;
    }


    for (int i = 0; i < a_size; i++)
    {
        if (agent_type[i] == 1) //Коренные жители
            glColor3ub(255, 0, 0); // Красный цвет
        if (agent_type[i] == 2) // Мигранты
            glColor3ub(0, 0, 255); // Синий цвет


        glRectd(x_a[i], y_a[i], x_a[i] + side_size, y_a[i] + side_size);
    }


    //redraw
    glutSwapBuffers();
    glutPostRedisplay();

}


void timer(int = 0)
{
    display();
    glutTimerFunc(1, timer, 0);
}


std::atomic<unsigned int> iterator0 = { 0 };
std::array<std::atomic_int, RUN_COUNT> myArray_id = {}; // Массив ID-потоков
FLAMEGPU_INIT_FUNCTION(init_function) {

    std::lock_guard<std::mutex> lock(m);
 
    // Извлечение значений переменных модели
    flamegpu::HostAgentAPI agent = FLAMEGPU->agent("agent");
    
    std::vector<Cells> high_tech_cells;
    std::vector<Cells> low_tech_cells;

    //Инициализация значений управляющих параметров модели (установка значений парамеров среды)

    double x = 0;
    double y = 0;

    std::vector<Cells> free_cells;

    int cells_count = 0;
    free_cells.clear();
    for (int j = 0; j < dim; j++)
    {
        x = 0;
        for (int i = 0; i < dim; i++)
        {
            if (occupied_cells[i][j] != 1)
                free_cells.push_back(Cells(x, y, i, j));

            x += side_size;
        }
        y += side_size;
    }

      
    //Генерация новых агентов-граждан
    
    int n = 0;
    if(free_cells.size() > 0)
    { 
        while (n < NumberOfCitizens && free_cells.size() > 0)
        {
            flamegpu::HostNewAgentAPI instance = agent.newAgent();
            int cells_count = free_cells.size();
            int indexx = 0;
            if (cells_count > 0)
                indexx = FLAMEGPU->random.uniform(0, cells_count - 1);
            else
                exit;

            double x1 = free_cells[indexx].x; // Строки
            double y1 = free_cells[indexx].y; // Столбцы

            int gender = (int)round(FLAMEGPU->random.uniform<double>());
            int age = (int)FLAMEGPU->random.uniform(1, 80);
            double education_level = (int)log(FLAMEGPU->random.logNormal<double>(2, 1));
            int language_level = 100;
            double comfort_level = round(FLAMEGPU->random.uniform(5, 10));
            int married = 0;
            int kids = 0;

            instance.setVariable<int>("id", getNextID());
            instance.setVariable<float>("x", free_cells[indexx].x);
            instance.setVariable<float>("y", free_cells[indexx].y);
            instance.setVariable<int>("row", free_cells[indexx].r);
            instance.setVariable<int>("clm", free_cells[indexx].c);
            instance.setVariable<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });
            instance.setVariable<int>("type", 1); // Граждане
            instance.setVariable<int>("current_state", 1);
            instance.setVariable<int>("previous_state", 1);
            instance.setVariable<int>("gender", gender);
            instance.setVariable<int>("age", age);
            instance.setVariable<double>("education_level", education_level);
            instance.setVariable<int>("language_level", language_level);
            instance.setVariable<double>("comfort_level", comfort_level);
            instance.setVariable<int>("married", married);
            instance.setVariable<int>("kids", kids);
            instance.setVariable<int>("type_resource", 0);
            instance.setVariable<int>("unemployed", 1);
            instance.setVariable<int>("t_arrival", 0);
           // instance.setVariable<int>("CI", -1);
            instance.setVariable<int>("move", 1);


            high_tech_cells.push_back(Cells(free_cells[indexx].x, free_cells[indexx].y, free_cells[indexx].r, free_cells[indexx].c));

            occupied_cells[free_cells[indexx].r][free_cells[indexx].c] = 1;
            if(indexx < free_cells.size())
            free_cells.erase(free_cells.begin() + indexx);
                      
      
            n++;
        }
    }
    //Генерация новых агентов-мигрантов
    
    n = 0;
    if(free_cells.size() > 0)
    { 
        while (n < NumberOfMigrants && free_cells.size() > 0)
        {
            flamegpu::HostNewAgentAPI instance = agent.newAgent();
            int cells_count = free_cells.size();
            int indexx = FLAMEGPU->random.uniform(0, cells_count - 1);
            double x1 = free_cells[indexx].x; // Строки
            double y1 = free_cells[indexx].y; // Столбцы

            int gender = (int)round(FLAMEGPU->random.uniform<double>());
            int age = (int)FLAMEGPU->random.uniform(1, 50);
            double education_level = 1;
            int language_level = 1;
            double comfort_level = round(FLAMEGPU->random.uniform(0, 5));
            int married = 0;
            int kids = 0;



            instance.setVariable<int>("id", getNextID());
            instance.setVariable<float>("x", free_cells[indexx].x);
            instance.setVariable<float>("y", free_cells[indexx].y);
            instance.setVariable<int>("row", free_cells[indexx].r);
            instance.setVariable<int>("clm", free_cells[indexx].c);
            instance.setVariable<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });
            instance.setVariable<int>("type", 2); // Мигранты
            instance.setVariable<int>("current_state", 1);
            instance.setVariable<int>("previous_state", 1);
            instance.setVariable<int>("gender", gender);
            instance.setVariable<int>("age", age);
            instance.setVariable<double>("education_level", education_level);
            instance.setVariable<int>("language_level", language_level);
            instance.setVariable<double>("comfort_level", comfort_level);
            instance.setVariable<int>("married", married);
            instance.setVariable<int>("kids", kids);
            instance.setVariable<int>("type_resource", 0);
            instance.setVariable<int>("unemployed", 1);
            instance.setVariable<int>("t_arrival", 0);
            //instance.setVariable<int>("CI", -1);
            instance.setVariable<int>("move", 1);

            low_tech_cells.push_back(Cells(free_cells[indexx].x, free_cells[indexx].y, free_cells[indexx].r, free_cells[indexx].c));

            occupied_cells[free_cells[indexx].r][free_cells[indexx].c] = 1;
            if (indexx < free_cells.size())
            free_cells.erase(free_cells.begin() + indexx);

            n++;
        }
    }
    //Генерация новых ресурсов
       // Извлечение значений переменных модели
    flamegpu::HostAgentAPI resource = FLAMEGPU->agent("resources");


    int cnt = 0;
    x = 0;
    y = 0;


    for (int j = 0; j < dim; j++)
    {
        x = 0;
        for (int i = 0; i < dim; i++)
        {
            flamegpu::HostNewAgentAPI instance = resource.newAgent();
            instance.setVariable<int>("id", getNextID()); // 
            instance.setVariable<float>("x", x); // 
            instance.setVariable<float>("y", y); // 
            instance.setVariable<int>("row", i); // 
            instance.setVariable<int>("clm", j); // 
            instance.setVariable<int>("type_resource", 0); // Клетка не имеет рабочего места
            instance.setVariable<int>("time_creation", 0); // Клетка не имеет рабочего места
            instance.setVariable<int>("is_occupied", 1); // По умолчанию все ячейки заняты
            instance.setVariable<unsigned int, 2>("pos", { (unsigned int)i, (unsigned int)j });
          //  instance.setVariable<int>("CI", -1); // Индекс кластера по умолчанию
            x += side_size;
            cnt++;
        }
        y += side_size;
    }
    //******************************************** КЛАСТЕРИЗАЦИЯ АГЕНТОВ (ЯЧЕЕК, ЗАНИМАЕМЫХ АГЕНТАМИ)************

        // Кластеризация коренных жителей *********************************************************************************
    std::vector<Cells>  centroids1;
    if (high_tech_cells.size() > 2)
    {
        // Выбор 3-х случайных агентов в качестве центров кластеров
        int a11 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
        int a21;
        do {
            a21 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
        } while (a21 == a11);
        int a31;
        do {
            a31 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
        } while (a31 == a11 || a31 == a21);


        // Добавление агентов в список центров кластеров
        centroids1.push_back(high_tech_cells.at(a11));
        centroids1.push_back(high_tech_cells.at(a21));
        centroids1.push_back(high_tech_cells.at(a31));

        //Вызов алгоритма кластеризации
        CMeansAlgorithm alg1(high_tech_cells, centroids1, 2);

        //Центры кластеров агентов-мигрантов
        float c11_x = alg1.getCentersX(0);
        float c11_y = alg1.getCentersY(0);
        float c21_x = alg1.getCentersX(1);
        float c21_y = alg1.getCentersY(1);
        float c31_x = alg1.getCentersX(2);
        float c31_y = alg1.getCentersY(2);

        // Вычисление строк и столбцов дискретного пространства являющихся центрами кластеров

        double d1_min = 100000;
        double d2_min = 100000;
        double d3_min = 100000;

        y = 0;
        for (int j = 0; j < dim; j++)
        {
            x = 0;
            for (int i = 0; i < dim; i++)
            {
                double d1 = sqrt(pow(c11_x - x, 2) + pow(c11_y - y, 2));
                double d2 = sqrt(pow(c21_x - x, 2) + pow(c21_y - y, 2));
                double d3 = sqrt(pow(c31_x - x, 2) + pow(c31_y - y, 2));

                if (d1 < d1_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_first_high_tech_cluster", { i, j });
                    d1_min = d1;
                }

                if (d2 < d2_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_second_high_tech_cluster", { i, j });
                    d2_min = d2;
                }

                if (d3 < d3_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_third_high_tech_cluster", { i, j });
                    d3_min = d3;
                }
                x += side_size;
            }
            y += side_size;
        }

        //Проверка 
        int r1 = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 0);
        int c1 = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 1);

        int r2 = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 0);
        int c2 = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 1);

        int r3 = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 0);
        int c3 = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 1);

    }

    // Кластеризация мигрантов *********************************************************************************
    std::vector<Cells>  centroids2;
    if (low_tech_cells.size() > 2)
    {
        // Выбор 3-х случайных агентов в качестве центров кластеров
        int a12 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
        int a22;
        do {
            a22 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
        } while (a22 == a12);
        int a32;
        do {
            a32 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
        } while (a32 == a12 || a32 == a22);


        // Добавление агентов в список центров кластеров
        centroids2.push_back(low_tech_cells.at(a12));
        centroids2.push_back(low_tech_cells.at(a22));
        centroids2.push_back(low_tech_cells.at(a32));

        //Вызов алгоритма кластеризации
        CMeansAlgorithm alg2(low_tech_cells, centroids2, 2);

        //Центры кластеров агентов-мигрантов
        float c12_x = alg2.getCentersX(0);
        float c12_y = alg2.getCentersY(0);
        float c22_x = alg2.getCentersX(1);
        float c22_y = alg2.getCentersY(1);
        float c32_x = alg2.getCentersX(2);
        float c32_y = alg2.getCentersY(2);

        // Вычисление строк и столбцов дискретного пространства являющихся центрами кластеров

        double d1_min = 100000;
        double d2_min = 100000;
        double d3_min = 100000;

        y = 0;
        for (int j = 0; j < dim; j++)
        {
            x = 0;
            for (int i = 0; i < dim; i++)
            {
                double d1 = sqrt(pow(c12_x - x, 2) + pow(c12_y - y, 2));
                double d2 = sqrt(pow(c22_x - x, 2) + pow(c22_y - y, 2));
                double d3 = sqrt(pow(c32_x - x, 2) + pow(c32_y - y, 2));

                if (d1 < d1_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_first_low_tech_cluster", { i, j });
                    d1_min = d1;
                }

                if (d2 < d2_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_second_low_tech_cluster", { i, j });
                    d2_min = d2;
                }

                if (d3 < d3_min)
                {
                    FLAMEGPU->environment.setProperty<int, 2>("Center_of_third_low_tech_cluster", { i, j });
                    d3_min = d3;
                }
                x += side_size;
            }
            y += side_size;
        }
    }

    //**********************************************************************************************************************
    int id_thr = std::hash<std::thread::id>{}(std::this_thread::get_id());
    myArray_id[iterator0] = id_thr; // ID потока
    iterator0++; // Индекс прогона
 }




FLAMEGPU_STEP_FUNCTION(cells_update) {
    std::lock_guard<std::mutex> lock(m);

    auto agent = FLAMEGPU->agent("agent");
    flamegpu::DeviceAgentVector population3 = agent.getPopulationData();

    for (int j = 0; j < dim; j++)
        for (int i = 0; i < dim; i++)
            occupied_cells[i][j] = 0;

    a_size = 0;
    std::vector<Cells> high_tech_cells;
    std::vector<Cells> low_tech_cells;
    high_tech_cells.clear();
    low_tech_cells.clear();

    std::vector<std::vector<int>> agent_in_cell;
    agent_in_cell.clear();
    agent_in_cell.assign(dim, std::vector<int>(dim));
       
    for (int i = 0; i < agent.count(); i++)
    {
        flamegpu::AgentVector::Agent instance = population3[i];

        int r = instance.getVariable<int>("row");
        int c = instance.getVariable<int>("clm");
        occupied_cells[r][c] = 1;

        x_a[i] = instance.getVariable<float>("x");
        y_a[i] = instance.getVariable<float>("y");
        agent_type[i] = instance.getVariable<int>("type");

        if (agent_type[i] == 1)
        {
            high_tech_cells.push_back(Cells(x_a[i], y_a[i], r, c));
            agent_in_cell[r][c] = 1;
        }
            
        if (agent_type[i] == 2)
        {
            low_tech_cells.push_back(Cells(x_a[i], y_a[i], r, c));
            agent_in_cell[r][c] = 2;
        }

        a_size++;
    }


    flamegpu::HostAgentAPI resource = FLAMEGPU->agent("resources");
    flamegpu::DeviceAgentVector population2 = resource.getPopulationData();

    for (int i = 0; i < resource.count(); i++)
    {
        flamegpu::AgentVector::Agent instance = population2[i];
        if (instance.getVariable<int>("id") != 0)
        {
            float x = instance.getVariable<float>("x");
            float y = instance.getVariable<float>("y");

            int r = instance.getVariable<int>("row");
            int c = instance.getVariable<int>("clm");
            resource_type[r][c] = instance.getVariable<int>("type_resource");
            int is_occupied = instance.getVariable<int>("is_occupied");
        }
    }
    
    //population2.syncChanges();
    population2.purgeCache();
    population3.purgeCache();

    //Выполнение кластеризации
    uint8_t Frequency = FLAMEGPU->environment.getProperty<uint8_t>("Frequency_work_places_creation");

    int year = FLAMEGPU->getStepCounter() + 1; // Год, предшествующий созданию рабочих мест
    if ( year % Frequency == 0 ) // 
    {
        //******************************************** КЛАСТЕРИЗАЦИЯ АГЕНТОВ (ЯЧЕЕК, ЗАНИМАЕМЫХ АГЕНТАМИ)************

    // Кластеризация коренных жителей *********************************************************************************
        std::vector<Cells>  centroids1;
        if (high_tech_cells.size() > 2)
        {
            // Выбор 3-х случайных агентов в качестве центров кластеров
            int a11 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
            int a21;
            do {
                a21 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
            } while (a21 == a11);
            int a31;
            do {
                a31 = (int)(FLAMEGPU->random.uniform<double>() * high_tech_cells.size());
            } while (a31 == a11 || a31 == a21);


            // Добавление агентов в список центров кластеров
            centroids1.push_back(high_tech_cells.at(a11));
            centroids1.push_back(high_tech_cells.at(a21));
            centroids1.push_back(high_tech_cells.at(a31));

            //Вызов алгоритма кластеризации
            CMeansAlgorithm alg1(high_tech_cells, centroids1, 2);

            //Центры кластеров агентов-мигрантов
            float c11_x = alg1.getCentersX(0);
            float c11_y = alg1.getCentersY(0);
            float c21_x = alg1.getCentersX(1);
            float c21_y = alg1.getCentersY(1);
            float c31_x = alg1.getCentersX(2);
            float c31_y = alg1.getCentersY(2);

            // Вычисление строк и столбцов дискретного пространства являющихся центрами кластеров

            double d1_min = 100000;
            double d2_min = 100000;
            double d3_min = 100000;

            double y = 0;
            for (int j = 0; j < dim; j++)
            {
                double x = 0;
                for (int i = 0; i < dim; i++)
                {
                    double d1 = sqrt(pow(c11_x - x, 2) + pow(c11_y - y, 2));
                    double d2 = sqrt(pow(c21_x - x, 2) + pow(c21_y - y, 2));
                    double d3 = sqrt(pow(c31_x - x, 2) + pow(c31_y - y, 2));

                    if (d1 < d1_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_first_high_tech_cluster", { i, j });
                        d1_min = d1;
                    }

                    if (d2 < d2_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_second_high_tech_cluster", { i, j });
                        d2_min = d2;
                    }

                    if (d3 < d3_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_third_high_tech_cluster", { i, j });
                        d3_min = d3;
                    }
                    x += side_size;
                }
                y += side_size;
            }

            //Проверка 
            int r1 = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 0);
            int c1 = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 1);

            int r2 = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 0);
            int c2 = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 1);

            int r3 = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 0);
            int c3 = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 1);

        }

        // Кластеризация мигрантов *********************************************************************************
        std::vector<Cells>  centroids2;
        if (low_tech_cells.size() > 2)
        {
            // Выбор 3-х случайных агентов в качестве центров кластеров
            int a12 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
            int a22;
            do {
                a22 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
            } while (a22 == a12);
            int a32;
            do {
                a32 = (int)(FLAMEGPU->random.uniform<double>() * low_tech_cells.size());
            } while (a32 == a12 || a32 == a22);


            // Добавление агентов в список центров кластеров
            centroids2.push_back(low_tech_cells.at(a12));
            centroids2.push_back(low_tech_cells.at(a22));
            centroids2.push_back(low_tech_cells.at(a32));

            //Вызов алгоритма кластеризации
            CMeansAlgorithm alg2(low_tech_cells, centroids2, 2);

            //Центры кластеров агентов-мигрантов
            float c12_x = alg2.getCentersX(0);
            float c12_y = alg2.getCentersY(0);
            float c22_x = alg2.getCentersX(1);
            float c22_y = alg2.getCentersY(1);
            float c32_x = alg2.getCentersX(2);
            float c32_y = alg2.getCentersY(2);

            // Вычисление строк и столбцов дискретного пространства являющихся центрами кластеров

            double d1_min = 100000;
            double d2_min = 100000;
            double d3_min = 100000;

            double y = 0;
            for (int j = 0; j < dim; j++)
            {
               double x = 0;
                for (int i = 0; i < dim; i++)
                {
                    double d1 = sqrt(pow(c12_x - x, 2) + pow(c12_y - y, 2));
                    double d2 = sqrt(pow(c22_x - x, 2) + pow(c22_y - y, 2));
                    double d3 = sqrt(pow(c32_x - x, 2) + pow(c32_y - y, 2));

                    if (d1 < d1_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_first_low_tech_cluster", { i, j });
                        d1_min = d1;
                    }

                    if (d2 < d2_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_second_low_tech_cluster", { i, j });
                        d2_min = d2;
                    }

                    if (d3 < d3_min)
                    {
                        FLAMEGPU->environment.setProperty<int, 2>("Center_of_third_low_tech_cluster", { i, j });
                        d3_min = d3;
                    }
                    x += side_size;
                }
                y += side_size;
            }
        }
        //*******************************************************************************************************
    }

    double DSI = Dunkan_segregation_index(high_tech_cells.size(), low_tech_cells.size(), agent_in_cell);

    FLAMEGPU->environment.setProperty<float>("Total_DSI", FLAMEGPU->environment.getProperty<float>("Total_DSI") + DSI);

    double DSI_avg = FLAMEGPU->environment.getProperty<float>("Total_DSI") / (FLAMEGPU->getStepCounter() + 1);
    FLAMEGPU->environment.setProperty<float>("Average_DSI", DSI_avg); // Среднее значение индекса DSI
}

std::array<std::atomic_int, RUN_COUNT> myArray_test1 = {};
std::array<std::atomic_int, RUN_COUNT> myArray_test2 = {};
std::array<std::atomic_int, RUN_COUNT> myArray_test3 = {};
FLAMEGPU_STEP_FUNCTION(BasicOutput) {

   std::lock_guard<std::mutex> lock(m);

    /* Выгрузка лога для ансамбля моделей */

    flamegpu::HostAgentAPI resource = FLAMEGPU->agent("resources");
    int id_thr = std::hash<std::thread::id>{}(std::this_thread::get_id());
    for(int i = 0 ; i < RUN_COUNT; i++)
    { 
       if(myArray_id[i]== id_thr)
                { 
                myArray_test1[i] = FLAMEGPU->getStepCounter(); // Обновляем счетчик времени
                myArray_test2[i] = FLAMEGPU->agent("agent").count(); // Количество агентов
                myArray_test3[i] = resource.count<int>("is_occupied", 0);// Количество свободных ячеек
                }
    }
   
   
    if (FLAMEGPU->getStepCounter() > 0)
    {
        flamegpu::HostAgentAPI resource = FLAMEGPU->agent("resources");
        // Извлечение значений переменных модели
        auto agent = FLAMEGPU->agent("agent");

        int count_free_cells = resource.count<int>("is_occupied", 0); // Количество свободных ячеек
        int count_occupied_cells = resource.count<int>("is_occupied", 1); // Количество несвободных ячеек
        int agents_count = agent.count(); // Количество агентов


        //auto location = FLAMEGPU->environment.getMacroProperty<uint32_t, 100, 100>("occupied_cells");

        double x = 0;
        double y = 0;

        std::vector<Cells> free_cells;

        int cells_count = 0;
        free_cells.clear();
        for (int j = 0; j < dim; j++)
        {
            x = 0;
            for (int i = 0; i < dim; i++)
            {
                if (occupied_cells[i][j] != 1) // Значит ячейка свободна
                    free_cells.push_back(Cells(x, y, i, j));

                x += side_size;
            }
            y += side_size;
        }

        int mig = agent.count<int>("type", 2);

        // Create NEW_AGENT_COUNT new 'agent' agents with 'x' set to 1.0f
        float share_of_new_migrants = FLAMEGPU->environment.getProperty<double>("Share_of_new_migrants");

        //Прибытие новых агентов-мигрантов
        

        int agent_migrants_count = agent.count<int>("type", 2);
        int agent_native_count = agent.count<int>("type", 1);
        int agent_zero_count = agent.count<int>("type", 0);
        int new_migrants = (int)round(agent_migrants_count * share_of_new_migrants);
        int n = 0;
      
        if(free_cells.size() > 0)
        { 
            while (n < new_migrants && free_cells.size() > 0) // 
            {
                //int check_agents =  agent.count<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });

                flamegpu::HostNewAgentAPI instance = agent.newAgent();

                int cells_count = free_cells.size();
                int indexx = FLAMEGPU->random.uniform(0, cells_count - 1);
                double x1 = free_cells[indexx].x; // Строки
                double y1 = free_cells[indexx].y; // Столбцы

                int gender = (int)round(FLAMEGPU->random.uniform<double>());
                int age = (int)FLAMEGPU->random.uniform(1, 50);
                double education_level = 1;
                int language_level = 1;
                double comfort_level = round(FLAMEGPU->random.uniform(0, 5));
                int married = 0;
                int kids = 0;


                instance.setVariable<int>("id", getNextID());
                instance.setVariable<float>("x", free_cells[indexx].x);
                instance.setVariable<float>("y", free_cells[indexx].y);
                instance.setVariable<int>("row", free_cells[indexx].r);
                instance.setVariable<int>("clm", free_cells[indexx].c);
                instance.setVariable<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });
                instance.setVariable<int>("type", 2); // Мигранты
                instance.setVariable<int>("current_state", 1);
                instance.setVariable<int>("previous_state", 1);
                instance.setVariable<int>("gender", gender);
                instance.setVariable<int>("age", age);
                instance.setVariable<double>("education_level", education_level);
                instance.setVariable<int>("language_level", language_level);
                instance.setVariable<double>("comfort_level", comfort_level);
                instance.setVariable<int>("married", married);
                instance.setVariable<int>("kids", kids);
                instance.setVariable<int>("type_resource", 0);
                instance.setVariable<int>("unemployed", 1);
                instance.setVariable<int>("t_arrival", FLAMEGPU->getStepCounter());
                //instance.setVariable<int>("CI", -1);
                instance.setVariable<int>("move", 1);

                occupied_cells[free_cells[indexx].r][free_cells[indexx].c] = 1;
                if (indexx < free_cells.size())
                free_cells.erase(free_cells.begin() + indexx);
            
                n++;
            }
        }
        //Рождение новых агентов-граждан
        agent_migrants_count = agent.count<int>("type", 2);
        agent_native_count = agent.count<int>("type", 1);
        agent_zero_count = agent.count<int>("type", 0);


        int number_of_natives_should_be_born = agent.count<int>("kids_should_be_born_native", 1);
               
        n = 0;
        if(free_cells.size() > 0)
        {
            while (n < number_of_natives_should_be_born && free_cells.size() > 0) // 
            {
                flamegpu::HostNewAgentAPI instance = agent.newAgent();

                int cells_count = free_cells.size();
                int indexx = 0;
                if (cells_count > 0)
                    indexx = FLAMEGPU->random.uniform(0, cells_count - 1);
                else
                    exit;

                double x1 = free_cells[indexx].x; // Строки
                double y1 = free_cells[indexx].y; // Столбцы

                int gender = (int)round(FLAMEGPU->random.uniform<double>());
                int age = 0;
                double education_level = 1;
                int language_level = 100;
                double comfort_level = 10;
                int married = 0;
                int kids = 0;


                instance.setVariable<int>("id", getNextID());
                instance.setVariable<float>("x", free_cells[indexx].x);
                instance.setVariable<float>("y", free_cells[indexx].y);
                instance.setVariable<int>("row", free_cells[indexx].r);
                instance.setVariable<int>("clm", free_cells[indexx].c);
                instance.setVariable<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });
                instance.setVariable<int>("type", 1); // Мигранты
                instance.setVariable<int>("current_state", 1);
                instance.setVariable<int>("previous_state", 1);
                instance.setVariable<int>("gender", gender);
                instance.setVariable<int>("age", age);
                instance.setVariable<double>("education_level", education_level);
                instance.setVariable<int>("language_level", language_level);
                instance.setVariable<double>("comfort_level", comfort_level);
                instance.setVariable<int>("married", married);
                instance.setVariable<int>("kids", kids);
                instance.setVariable<int>("type_resource", 0);
                instance.setVariable<int>("unemployed", 1);
                instance.setVariable<int>("t_arrival", FLAMEGPU->getStepCounter());
              //  instance.setVariable<int>("CI", -1);
                instance.setVariable<int>("move", 1);

                occupied_cells[free_cells[indexx].r][free_cells[indexx].c] = 1;
                if (indexx < free_cells.size())
                free_cells.erase(free_cells.begin() + indexx);

                n++;
            }
        }

        //Рождение новых агентов-мигрантов
        agent_migrants_count = agent.count<int>("type", 2);
        agent_native_count = agent.count<int>("type", 1);
        agent_zero_count = agent.count<int>("type", 0);

        int number_of_migrants_should_be_born = agent.count<int>("kids_should_be_born_migrant", 1);

        n = 0;
        if(free_cells.size() > 0)
        { 
            while (n < number_of_migrants_should_be_born && free_cells.size() > 0) // 
            {
                flamegpu::HostNewAgentAPI instance = agent.newAgent();

                int cells_count = free_cells.size();
                int indexx = 0;
                if(cells_count > 0)
                    indexx = FLAMEGPU->random.uniform(0, cells_count - 1);
                else
                    exit;

                double x1 = free_cells[indexx].x; // Строки
                double y1 = free_cells[indexx].y; // Столбцы

                int gender = (int)round(FLAMEGPU->random.uniform<double>());
                int age = 0;
                double education_level = 0;
                int language_level = 0;
                double comfort_level = 10;
                int married = 0;
                int kids = 0;


                instance.setVariable<int>("id", getNextID());
                instance.setVariable<float>("x", free_cells[indexx].x);
                instance.setVariable<float>("y", free_cells[indexx].y);
                instance.setVariable<int>("row", free_cells[indexx].r);
                instance.setVariable<int>("clm", free_cells[indexx].c);
                instance.setVariable<unsigned int, 2>("pos", { (unsigned int)free_cells[indexx].r, (unsigned int)free_cells[indexx].c });
                instance.setVariable<int>("type", 2); // Мигранты
                instance.setVariable<int>("current_state", 1);
                instance.setVariable<int>("previous_state", 1);
                instance.setVariable<int>("gender", gender);
                instance.setVariable<int>("age", age);
                instance.setVariable<double>("education_level", education_level);
                instance.setVariable<int>("language_level", language_level);
                instance.setVariable<double>("comfort_level", comfort_level);
                instance.setVariable<int>("married", married);
                instance.setVariable<int>("kids", kids);
                instance.setVariable<int>("type_resource", 0);
                instance.setVariable<int>("unemployed", 1);
                instance.setVariable<int>("t_arrival", FLAMEGPU->getStepCounter());
             //   instance.setVariable<int>("CI", -1);
                instance.setVariable<int>("move", 1);

                occupied_cells[free_cells[indexx].r][free_cells[indexx].c] = 1;
                if(indexx < free_cells.size())
                free_cells.erase(free_cells.begin() + indexx);

                n++;
            }
        }
        // Расчет отдельных макроэкономических показателей

        //Цены и Параметры ПФКД для высокотехнологичного ресурса, занятого Агентом
        double p1 = 1000;  //Средняя цена на продукцию высокотехноличных отраслей экономики

        double A_r1 = 1.01;
        double a_r1 = 0.1;
        double b_r1 = 0.9;


        double G_r1 = 0.1;
        double c_r1 = 0.1;
        double d_r1 = 0.9;


        //Цены и Параметры ПФКД для примитивного ресурса, занятого Агентом
        double p2 = 300; //Средняя цена на продукцию низкокотехноличных отраслей экономики

        double A_r2 = 1;
        double a_r2 = 0.2;
        double b_r2 = 0.8;

        double G_r2 = 1.5;
        double c_r2 = 0.5;
        double d_r2 = 0.5;

        int agents_in_high_sectors = agent.count<int>("type_resource", 2);
        int agents_in_low_sectors = agent.count<int>("type_resource", 1);
        int unemployment_agents = agent.count<int>("unemployed", 1);
        int agent_pensioners = agent.count<int>("is_pensioner", 1);
        int number_of_assimilated_agents = agent.count<int>("is_assimilated", 1);

        float Expenditure_on_pensioners = 2400; // условных единиц пенсии в год на 1 пенсионера
        float Expenditure_on_unemploymenters = 2000; // условных единиц пенсии в год на 1 безработного

        // 1.2 и 0.8 это уровень производственной отдачи в высокотехнологичном и низкотехнологичных отраслях экономики
        float V = p1 * (1.2 * agents_in_high_sectors) + p2 * (0.8 * agents_in_low_sectors) -
            agent_pensioners * Expenditure_on_pensioners -
            unemployment_agents * Expenditure_on_unemploymenters;

        float GE =
            (agent_pensioners * Expenditure_on_pensioners +
                unemployment_agents * Expenditure_on_unemploymenters +
                agents_in_low_sectors * 12000) / 1000000; // 12000 - стоимость создания 1 рабочего места для мигрантов




        float Rate_V = 0;
        float Rate_GE = 0;

        if (FLAMEGPU->getStepCounter() == 1)
        {
            Rate_V = 1.05;
            Rate_GE = 1.01;
        }

        else
        {
            Rate_V = V / FLAMEGPU->environment.getProperty<float>("GDP"); // ВВП
            Rate_GE = GE / FLAMEGPU->environment.getProperty<float>("GE"); //Гос. расходы
        }

        Rate_V += FLAMEGPU->environment.getProperty<float>("GDP_rate_total");
        Rate_GE += FLAMEGPU->environment.getProperty<float>("GE_rate_total");
        FLAMEGPU->environment.setProperty<float>("GDP_rate_total", Rate_V);
        FLAMEGPU->environment.setProperty<float>("GE_rate_total", Rate_GE);

        float average_Rate_GDP = FLAMEGPU->environment.getProperty<float>("GDP_rate_total") / FLAMEGPU->getStepCounter(); // Средний темп роста ВВП
        float average_Rate_GE = FLAMEGPU->environment.getProperty<float>("GE_rate_total") / FLAMEGPU->getStepCounter(); // Средний темп роста Государственных расходов

        FLAMEGPU->environment.setProperty<float>("Average_GDP_rate", average_Rate_GDP);
        FLAMEGPU->environment.setProperty<float>("Average_Government_Expenditure_rate", average_Rate_GE);

        FLAMEGPU->environment.setProperty<float>("GDP", V);
        FLAMEGPU->environment.setProperty<float>("GE", GE);

    }


    if (out.is_open() && VIS_MODE == 1) // Выгрузка результатов для одиночного прогона
    {
        if (FLAMEGPU->getStepCounter() == 0)
        {
            out << "Share_of_new_migrants" <<
                ";" << "Expenditure_on_education_share" <<
                ";" << "Life_age_high_technology_work_places" <<
                ";" << "Life_age_low_technology_work_places" <<
                ";" << "Frequency_work_places_creation" <<
                ";" << "Average_life_time_of_natives" <<
                ";" << "Average_life_time_of_migrants" <<
                ";" << "Age_for_married_and_kids_birth_of_natives" <<
                ";" << "Age_for_married_and_kids_birth_of_migrants" <<
                ";" << "Minimum_comfort_level_of_natives" <<
                ";" << "Minimum_comfort_level_of_migrants" <<
                ";" << "Pension_age" <<
                ";" << "Method_work_places_creation" <<

                ";" << "Total_count_of_agents" <<
                ";" << "Share_of_non-assimilated_migrants" <<
                ";" << "Number_of_assimilated_migrants" <<
                ";" << "Averaged_time_for_assimilation" <<
                ";" << "Duncan_Segregation_Index" <<
                ";" << "Average_GDP_rate" <<
                ";" << "Average_Government_Expenditure_rate" << std::endl;
        }

        //Выгрузка полученных результатов в динамике
        if (FLAMEGPU->getStepCounter() > 0)
        {
            auto agent = FLAMEGPU->agent("agent");
            
            int agent_count = 0;
            if (agent.count() > 0)
                agent_count = agent.count();

            float t_assim = 0;
            if (agent.count<int>("type", 2) > 0)
                t_assim = (double)(agent.sum<double>("Time_for_assimilation") / agent.count<int>("type", 2));

            int number_of_nonassimilated = agent.count<int>("type", 2);
            float share_of_nonassimilated = (float)number_of_nonassimilated / agent_count;


            out << FLAMEGPU->environment.getProperty<double>("Share_of_new_migrants") <<
                ";" << FLAMEGPU->environment.getProperty<double>("Expenditure_on_education_share") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Life_age_high_technology_work_places") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Life_age_low_technology_work_places") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Frequency_work_places_creation") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_natives") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_migrants") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_natives") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_migrants") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_natives") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_migrants") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Pension_age") <<
                ";" << (int)FLAMEGPU->environment.getProperty<uint8_t>("Method_work_places_creation") <<

                ";" << agent.count() <<
                ";" << share_of_nonassimilated <<
                ";" << agent.count<int>("is_assimilated", 1) <<
                ";" << t_assim <<
                ";" << FLAMEGPU->environment.getProperty<float>("Average_DSI") <<
                ";" << FLAMEGPU->environment.getProperty<float>("Average_GDP_rate") <<
                ";" << FLAMEGPU->environment.getProperty<float>("Average_Government_Expenditure_rate") << std::endl;
        }

        if (FLAMEGPU->getStepCounter() == TIME_STOP - 1)
            out.close();

    }

}


std::array<std::atomic_int, RUN_COUNT> myArray1 = {};
std::array<std::atomic_int, RUN_COUNT> myArray2 = {};
std::array<std::atomic_int, RUN_COUNT> myArray3 = {};
std::array<std::atomic_int, RUN_COUNT> myArray4 = {};
std::array<std::atomic_int, RUN_COUNT> myArray5 = {};
std::array<std::atomic_int, RUN_COUNT> myArray6 = {};
std::array<std::atomic_int, RUN_COUNT> myArray7 = {};
std::array<std::atomic_int, RUN_COUNT> myArray8 = {};
std::array<std::atomic_int, RUN_COUNT> myArray9 = {};
std::array<std::atomic_int, RUN_COUNT> myArray10 = {};
std::array<std::atomic_int, RUN_COUNT> myArray11 = {};
std::array<std::atomic_int, RUN_COUNT> myArray12 = {};
std::array<std::atomic_int, RUN_COUNT> myArray13 = {};
std::array<std::atomic_int, RUN_COUNT> myArray14 = {};
std::array<std::atomic_int, RUN_COUNT> myArray15 = {};
std::array<std::atomic_int, RUN_COUNT> myArray16 = {};
std::array<std::atomic_int, RUN_COUNT> myArray17 = {};
std::array<std::atomic_int, RUN_COUNT> myArray18 = {};
std::array<std::atomic_int, RUN_COUNT> myArray19 = {};
std::array<std::atomic_int, RUN_COUNT> myArray20 = {};

std::atomic<unsigned int> iterator = { 0 };
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    std::lock_guard<std::mutex> lock(m);

	if (FLAMEGPU->getStepCounter() >= TIME_STOP - 1)
	{
		
		auto agent = FLAMEGPU->agent("agent");

		flamegpu::DeviceAgentVector population(agent.getPopulationData());
		a_size = 0;
		int counter = 0;
		double assim_t = 0.0;

		for (int i = 0; i < agent.count(); i++)
		{
			flamegpu::AgentVector::Agent instance = population[i];
			x_a[i] = instance.getVariable<float>("x");
			y_a[i] = instance.getVariable<float>("y");
			agent_type[i] = instance.getVariable<int>("type");

			if (instance.getVariable<double>("Time_for_assimilation") > 0)
			{
				assim_t += instance.getVariable<double>("Time_for_assimilation");
				counter++;
			}

			a_size++;
		}



		int agent_migrants_count = agent.count<int>("type", 2); // Количество мигрантов
		float migrants_share = (float)agent_migrants_count / agent.count(); // Доля мигрантов

		float average_assim_time = 0;
		if (agent_migrants_count > 0)
			average_assim_time = (double)agent.sum<double>("Time_for_assimilation") / agent_migrants_count; // Среднее врем на ассимиляцию

		if (counter > 0)
			assim_t = assim_t / counter; //  Среднее время на ассимиляцию (уточненный расчет)

		population.syncChanges();
		population.purgeCache();
		
		
		myArray1[iterator] = (int)(FLAMEGPU->environment.getProperty<double>("Share_of_new_migrants") * 100000);
		myArray2[iterator] = (int)(FLAMEGPU->environment.getProperty<double>("Expenditure_on_education_share") * 100000);
		myArray3[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Life_age_high_technology_work_places");
		myArray4[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Life_age_low_technology_work_places");
		myArray5[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Frequency_work_places_creation");
		myArray6[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_natives");
		myArray7[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_migrants");
        myArray8[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_natives");
		myArray9[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_migrants");
		myArray10[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_natives");
		myArray11[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_migrants");
		myArray12[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Pension_age");
        myArray13[iterator] = FLAMEGPU->environment.getProperty<uint8_t>("Method_work_places_creation");
        
        myArray14[iterator] = agent.count();
        myArray15[iterator] = (int)(migrants_share*100000);
        myArray16[iterator] = agent.count<int>("is_assimilated", 1);;
        myArray17[iterator] = (int)(assim_t * 100000);
        myArray18[iterator] = (int)(FLAMEGPU->environment.getProperty<float>("Average_DSI") * 100000);
        myArray19[iterator] = (int)(FLAMEGPU->environment.getProperty<float>("Average_GDP_rate") * 100000);
        myArray20[iterator] = (int)(FLAMEGPU->environment.getProperty<float>("Average_Government_Expenditure_rate") * 100000);

		iterator++;
		return  flamegpu::EXIT;  // End the simulation here
	}
    else
        return  flamegpu::CONTINUE;  // Continue the simulation
}


FLAMEGPU_AGENT_FUNCTION(all_agents, flamegpu::MessageNone, flamegpu::MessageArray2D)
{
    //Передача данных о каждом имеющимся агенте
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<int>("row", FLAMEGPU->getVariable<int>("row"));
    FLAMEGPU->message_out.setVariable<int>("clm", FLAMEGPU->getVariable<int>("clm"));
    FLAMEGPU->message_out.setVariable<int>("move", FLAMEGPU->getVariable<int>("move"));
    FLAMEGPU->message_out.setVariable<int>("current_state", FLAMEGPU->getVariable<int>("current_state"));
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
    FLAMEGPU->message_out.setVariable<int>("married", FLAMEGPU->getVariable<int>("married"));
    FLAMEGPU->message_out.setVariable<int>("gender", FLAMEGPU->getVariable<int>("gender"));
    FLAMEGPU->message_out.setVariable<double>("education_level", FLAMEGPU->getVariable<double>("education_level"));

    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));

    return  flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(all_resources, flamegpu::MessageNone, flamegpu::MessageArray2D)
{
    //Передача данных о каждом ресурсе
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<int>("type_resource", FLAMEGPU->getVariable<int>("type_resource"));
    FLAMEGPU->message_out.setVariable<int>("is_occupied", FLAMEGPU->getVariable<int>("is_occupied"));
    FLAMEGPU->message_out.setVariable<int>("row", FLAMEGPU->getVariable<int>("row"));
    FLAMEGPU->message_out.setVariable<int>("clm", FLAMEGPU->getVariable<int>("clm"));
    FLAMEGPU->message_out.setVariable<int>("request_agent", FLAMEGPU->getVariable<int>("request_agent"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    
    return  flamegpu::ALIVE;
}


//Проверка наличия "двойных" агентов
FLAMEGPU_AGENT_FUNCTION(check_all_agents, flamegpu::MessageArray2D, flamegpu::MessageNone)
{
    int flag = 0;

    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    //Клетка 1x1
    // Значит сообщение пришло от агента, расположенного в тех же координатах
    const auto message = FLAMEGPU->message_in.at(my_x, my_y); 
    
        if (message.getVariable<int>("row") == FLAMEGPU->getVariable<int>("row") &&
            message.getVariable<int>("clm") == FLAMEGPU->getVariable<int>("clm") &&
            message.getVariable<int>("id") != FLAMEGPU->getVariable<int>("id"))
        {
            flag = 1; // Обнаружен "двойной" агент
        }
    
    if (flag == 0)
        return  flamegpu::ALIVE;
    else
    {
        printf("Agent with ID is %i is lost\n", FLAMEGPU->getVariable<int>("id"));
        return  flamegpu::DEAD;
    }
}


FLAMEGPU_AGENT_FUNCTION(workplaces_creation, flamegpu::MessageArray2D, flamegpu::MessageNone)
{

    uint8_t method = FLAMEGPU->environment.getProperty<uint8_t>("Method_work_places_creation");
    uint8_t Frequency = FLAMEGPU->environment.getProperty<uint8_t>("Frequency_work_places_creation");
    unsigned int my_x = 0;
    unsigned int my_y = 0;



    if (method == 1 && FLAMEGPU->getStepCounter() % Frequency == 0) // Равномерное и смешанное создание рабочих мест c заданной периодичностью
    {
        double p = FLAMEGPU->random.uniform<double>();

        //Создание различных рабочих мест с некоторой вероятностью
        if ((FLAMEGPU->getVariable<int>("is_occupied") == 0 || FLAMEGPU->getStepCounter() == 0) &&
            FLAMEGPU->getVariable<int>("type_resource") == 0 && p > 0.7)
        {
            FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        else if ((FLAMEGPU->getVariable<int>("is_occupied") == 0 || FLAMEGPU->getStepCounter() == 0) &&
            FLAMEGPU->getVariable<int>("type_resource") == 0 &&
            p > 0.6 && p <= 0.7)
        {
            FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // 
        }
    }

    if (method == 2 && FLAMEGPU->getStepCounter() % Frequency == 0) // Кластерно-ориентированное создание новых рабочих мест
    {
       // Высокотехнологичные рабочие места ***************************************************************************

        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_first_high_tech_cluster", 1);
        int S1 = FLAMEGPU->environment.getProperty<int>("Size_of_high_tech_cluster"); // Размерность кластеров-рабочих мест
        int S2 = FLAMEGPU->environment.getProperty<int>("Size_of_low_tech_cluster");

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }
       
        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1)) 
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }


        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_second_high_tech_cluster", 1);

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1))
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }


        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_third_high_tech_cluster", 1);

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1))
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 2); // Новый высокотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }


        // Низкотехнологичные рабочие места **********************************************************************
        
        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_first_low_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_first_low_tech_cluster", 1);

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1))
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }

        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_second_low_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_second_low_tech_cluster", 1);

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1))
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }

        my_x = FLAMEGPU->environment.getProperty<int>("Center_of_third_low_tech_cluster", 0);  // Координаты центра кластера
        my_y = FLAMEGPU->environment.getProperty<int>("Center_of_third_low_tech_cluster", 1);

        if (FLAMEGPU->getVariable<int>("row") == my_x && FLAMEGPU->getVariable<int>("clm") == my_y)
        {
            FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
            FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
        }

        for (const auto& message : FLAMEGPU->message_in.wrap(FLAMEGPU->getVariable<int>("row"), FLAMEGPU->getVariable<int>("clm"), S1))
        {
            if (message.getVariable<int>("row") == my_x && message.getVariable<int>("clm") == my_y)
            {
                FLAMEGPU->setVariable<int>("type_resource", 1); // Новый низкотехнологичный ресурс
                FLAMEGPU->setVariable<int>("time_creation", FLAMEGPU->getStepCounter()); // Время создания ресурса
            }
        }

    }

    // Выбытие рабочих мест при условии достижения порогового уровня возраста
    uint8_t age_max_high = FLAMEGPU->environment.getProperty<uint8_t>("Life_age_high_technology_work_places");
    uint8_t age_max_low = FLAMEGPU->environment.getProperty<uint8_t>("Life_age_low_technology_work_places");

    if (FLAMEGPU->getStepCounter() - FLAMEGPU->getVariable<int>("time_creation") >= age_max_high && FLAMEGPU->getVariable<int>("type_resource") == 2)
    {
        FLAMEGPU->setVariable<int>("type_resource", 0);
        FLAMEGPU->setVariable<int>("time_creation", 0);
    }

    if (FLAMEGPU->getStepCounter() - FLAMEGPU->getVariable<int>("time_creation") >= age_max_low && FLAMEGPU->getVariable<int>("type_resource") == 1)
    {
        FLAMEGPU->setVariable<int>("type_resource", 0);
        FLAMEGPU->setVariable<int>("time_creation", 0);
    }
    return  flamegpu::ALIVE;
}


//Функция обновления состояния ячеёки (ресурса)
FLAMEGPU_AGENT_FUNCTION(update_cell, flamegpu::MessageArray2D, flamegpu::MessageNone) {

    int flag = 0;

    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    FLAMEGPU->setVariable<int>("is_occupied", 0); // По умолчанию все клетки свободны

    const auto message = FLAMEGPU->message_in.at(my_x, my_y); 
        
    if ((message.getVariable<int>("row") == FLAMEGPU->getVariable<int>("row") &&
                message.getVariable<int>("clm") == FLAMEGPU->getVariable<int>("clm")))
            {
                flag = 1;
            }
       
   // auto swap = FLAMEGPU->environment.getMacroProperty<uint32_t, 100, 100>("occupied_cells");
   
    if (flag == 0 && FLAMEGPU->getVariable<int>("waiting_occupation") != 1)
    {
        FLAMEGPU->setVariable<int>("is_occupied", 0);
    //    swap[my_x][my_y].exchange(0);
    }
    if (flag == 1)
    {
        FLAMEGPU->setVariable<int>("is_occupied", 1);
    //    swap[my_x][my_y].exchange(1);
    }


    return  flamegpu::ALIVE;
}

//Функция проверки того, что агент занимает данную ячейку
FLAMEGPU_AGENT_FUNCTION(check_cell, flamegpu::MessageArray2D, flamegpu::MessageNone) {

    unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);


    const auto message = FLAMEGPU->message_in.at(my_x, my_y);

        if (message.getVariable<int>("row") == FLAMEGPU->getVariable<int>("row") &&
            message.getVariable<int>("clm") == FLAMEGPU->getVariable<int>("clm"))
        {
            //Значит агент занимает эту ячейку
            FLAMEGPU->setVariable<int>("type_resource", message.getVariable<int>("type_resource"));


            if (FLAMEGPU->getVariable<int>("type_resource") == 0)
                FLAMEGPU->setVariable<int>("unemployed", 1); // Агент не имеет работы


            if (FLAMEGPU->getVariable<int>("type_resource") != 0)
                FLAMEGPU->setVariable<int>("unemployed", 0); // Агент имеет работу

        }

    return  flamegpu::ALIVE;
}



//Функция оценки количество взаимодействий местный житель-мигрант
FLAMEGPU_AGENT_FUNCTION(agent_to_agent_contacts, flamegpu::MessageArray2D, flamegpu::MessageNone) {


    unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    int my_type = FLAMEGPU->getVariable<int>("type");


    //Мурово-соседство
    for (const auto& message : FLAMEGPU->message_in(my_x, my_y))
    {

        int r1 = message.getVariable<int>("row");
        int c1 = message.getVariable<int>("clm");
        int r2 = FLAMEGPU->getVariable<int>("row");
        int c2 = FLAMEGPU->getVariable<int>("clm");

        int type_m = message.getVariable<int>("type");

        // Мурово-соседство
        if (((r1 == r2 - 1 || r1 == r2 + 1) && c1 == c2 ||
            (r1 == r2 - 1 || r1 == r2 + 1) && c1 == c2 - 1 ||
            (r1 == r2 - 1 || r1 == r2 + 1) && c1 == c2 + 1) ||

            ((c1 == c2 - 1 || c1 == c2 + 1) && r1 == r2 ||
                (c1 == c2 - 1 || c1 == c2 + 1) && r1 == r2 - 1 ||
                (c1 == c2 - 1 || c1 == c2 + 1) && r1 == r2 + 1))
        {
            //Контакты агента-гражданина с агентами-мигрантами
            if (my_type == 1 && type_m == 2)
                FLAMEGPU->setVariable<double>("comfort_level", FLAMEGPU->getVariable<double>("comfort_level") - 0.1); // Небольшое снижение уровня личного комфорта при контакте с мигрантами 

            //Контакты агента-мигранта с агентами-гражданами
            if (my_type == 2 && type_m == 1)
                FLAMEGPU->setVariable<int>("language_level", FLAMEGPU->getVariable<int>("language_level") + 1); // Улучшение уровня языковых навыков
        }
    }

    return  flamegpu::ALIVE;
}


FLAMEGPU_AGENT_FUNCTION(looking_for_partner, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {


    unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    int my_state = FLAMEGPU->getVariable<int>("current_state");
    int my_gender = FLAMEGPU->getVariable<int>("gender");
    int my_type = FLAMEGPU->getVariable<float>("type");
    int my_married = FLAMEGPU->getVariable<int>("married");

    double max_distance = 100000000;
    int flag = 0;
    int ID_partner = 0;

    for (const auto& message : FLAMEGPU->message_in.wrap(my_x, my_y, 10)) // 10 ячеек радиус поиска (49 - макс. при размерности 100x100)
    {
        int ID = message.getVariable<int>("id");
        float x_m = message.getVariable<float>("x");
        float y_m = message.getVariable<float>("y");
        int state = message.getVariable<int>("current_state");
        int gender = message.getVariable<float>("gender");
        int type = message.getVariable<float>("type");
        int married = message.getVariable<int>("married");


        double distance = sqrt(pow(x_m - FLAMEGPU->getVariable<float>("x"), 2) + pow(y_m - message.getVariable<float>("y"), 2)); // Расстояние до агента партнера

        if (my_gender != gender &&
            my_married == 0 && married == 0 &&
            my_state == 3 && state == 3 && distance < max_distance) // Если оба агента находятся в состоянии поиска партнера
        {
            flag = 1;
            ID_partner = ID;
            max_distance = distance;
        }
    }

    if (flag == 1) // Партнер найден и должна быть установлена взаимная связь
    {
        FLAMEGPU->setVariable<int>("married", 1);
        FLAMEGPU->setVariable<int>("ID_partner", ID_partner);
        // Отправка сообщения агенту-партнеру для установления взаимной (парной) связи
        FLAMEGPU->message_out.setVariable<int>("ID_partner", FLAMEGPU->getVariable<int>("id")); // Собственный ID
        FLAMEGPU->message_out.setVariable<int>("ID_married", ID_partner); // ID агента с которым заключается брак 


        FLAMEGPU->setVariable<int>("move", 0); // Переход агента в стационарное состояние
        FLAMEGPU->setVariable<int>("current_state", 1); // Переход агента в стационарное состояние

    }

    if (flag == 0) // Партнер не найден
    {
        FLAMEGPU->message_out.setVariable<int>("ID_partner", 0); // Собственный ID
        FLAMEGPU->message_out.setVariable<int>("ID_married", 0); // ID агента с которым заключается брак 
        FLAMEGPU->setVariable<int>("move", 1); // Перемещение агента в пространстве
    }


    FLAMEGPU->message_out.setIndex(my_x, my_y);

    return  flamegpu::ALIVE;
}


FLAMEGPU_AGENT_FUNCTION(looking_for_resource, flamegpu::MessageArray2D, flamegpu::MessageArray2D) {

    unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0); // Координаты ресурса
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    int type_resource = FLAMEGPU->getVariable<int>("type_resource"); // Тип ресурса
    int is_occupied = FLAMEGPU->getVariable<int>("is_occupied");// Занят ли ресурс

    double max_distance = 100000000;
    int flag = 0;
    int ID_partner = 0;
    int r_res = 0;
    int c_res = 0;
    float x_res = 0;
    float y_res = 0;
    int ID_resource = 0;
    int ID_agent = 0;

    FLAMEGPU->setVariable("waiting_occupation", 0);

    int S_min = 10; // Минимальная размерность области поиска рабочего места
    float educ = FLAMEGPU->environment.getProperty<float>("Influence_of_education_on_employment_opportunities");

    for (const auto& message : FLAMEGPU->message_in(my_x, my_y, (dim/4)-1)) // 49 - макс. при размерности 100x100
    {
        float x_m = message.getVariable<float>("x"); // Координаты агента
        float y_m = message.getVariable<float>("y");

        int my_state = message.getVariable<int>("current_state"); // Состояние агента
        int my_type = message.getVariable<int>("type"); // Тип агента

        double distance = sqrt(pow(x_m - FLAMEGPU->getVariable<float>("x"), 2) + pow(y_m - FLAMEGPU->getVariable<float>("y"), 2)); // Расстояние до агента

        max_distance = 10 * side_size * (1 + message.getVariable<double>("education_level") * educ);

      
        if (my_state == 2 && is_occupied == 0 && // Если агент находится в состоянии поиска рабочего места
            ((my_type == 1 && type_resource == 2) || (my_type == 2 && type_resource == 1)) && distance < max_distance && is_occupied == 0)
        {
            ID_agent = message.getVariable<int>("id"); // ID Агента
            ID_resource = FLAMEGPU->getVariable<int>("id"); // ID Ресурса
            x_res = FLAMEGPU->getVariable<float>("x");
            y_res = FLAMEGPU->getVariable<float>("y");
            r_res = FLAMEGPU->getVariable<int>("row");
            c_res = FLAMEGPU->getVariable<int>("clm");
            FLAMEGPU->setVariable<int>("waiting_occupation", 1); // Ожидание занятия ресурса каким-либо агентом
            FLAMEGPU->setVariable<int>("is_occupied", 1);
            flag = 1;
            max_distance = distance;

            FLAMEGPU->message_out.setVariable<float>("target_x", x_res);
            FLAMEGPU->message_out.setVariable<float>("target_y", y_res);
            FLAMEGPU->message_out.setVariable<int>("target_row", r_res);
            FLAMEGPU->message_out.setVariable<int>("target_clm", c_res);

        }
    }


    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<int>("row", FLAMEGPU->getVariable<int>("row"));
    FLAMEGPU->message_out.setVariable<int>("clm", FLAMEGPU->getVariable<int>("clm"));

    FLAMEGPU->message_out.setVariable<int>("agent_ID", ID_agent); // ID агента
    FLAMEGPU->message_out.setVariable<int>("target_ID", ID_resource); // ID ресурса, который должен быть занят

    FLAMEGPU->message_out.setIndex(my_x, my_y);

    return flamegpu::ALIVE;
}


FLAMEGPU_AGENT_FUNCTION(moving_trasaction, flamegpu::MessageArray2D, flamegpu::MessageNone)
{

    unsigned int my_x = FLAMEGPU->getVariable<unsigned  int, 2>("pos", 0);
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);
    int my_ID = FLAMEGPU->getVariable<int>("id");

    for (const auto& message : FLAMEGPU->message_in.wrap(my_x, my_y, 49)) // 49 максимальный радиус соседства при размерности 100 x 100
    {
        if (my_ID == message.getVariable<int>("agent_ID") && message.getVariable<int>("target_ID") != 0) // Если передан не нулевой id_ресурса
        {
            //   printf("%i, %i, moving_trasaction is run \n", message.getVariable<int>("target_ID"));

            float x_res = message.getVariable<float>("target_x");  // Координаты целевого ресурса
            float y_res = message.getVariable<float>("target_y");
            int r_res = message.getVariable<int>("target_row");
            int c_res = message.getVariable<int>("target_clm");

            FLAMEGPU->setVariable<float>("x", x_res);
            FLAMEGPU->setVariable<float>("y", y_res);

            FLAMEGPU->setVariable<int>("row", r_res);
            FLAMEGPU->setVariable<int>("clm", c_res);

            FLAMEGPU->setVariable<unsigned int, 2>("pos", 0, r_res);
            FLAMEGPU->setVariable<unsigned int, 2>("pos", 1, c_res);

            FLAMEGPU->setVariable<int>("move", 0); // Переход агента в стационарное состояние
            FLAMEGPU->setVariable<int>("current_state", 1); // Переход агента в стационарное состояние
        }
    }

    return flamegpu::ALIVE;

}



FLAMEGPU_AGENT_FUNCTION(get_married, flamegpu::MessageArray2D, flamegpu::MessageNone) {

    unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    int my_ID = FLAMEGPU->getVariable<int>("id");
    for (const auto& message : FLAMEGPU->message_in(my_x, my_y, dim))
    {
        int ID_partner = message.getVariable<int>("ID_partner");
        int ID_married = message.getVariable<int>("ID_married");

        if (my_ID == ID_married)
        {
            FLAMEGPU->setVariable<int>("ID_partner", ID_partner); // Связь с партнером установлена
            FLAMEGPU->setVariable<int>("married", 1); // Связь с партнером установлена
        }
    }
    return flamegpu::ALIVE;
}




FLAMEGPU_AGENT_FUNCTION(update_agent_state, flamegpu::MessageNone, flamegpu::MessageNone) {


    int current_age = FLAMEGPU->getVariable<int>("age") + FLAMEGPU->getStepCounter() - FLAMEGPU->getVariable<int>("t_arrival");
    double comfort_level = FLAMEGPU->getVariable<double>("comfort_level");
    int type = FLAMEGPU->getVariable<int>("type");
    int married = FLAMEGPU->getVariable<int>("married");
    int kids = FLAMEGPU->getVariable<int>("kids");
    int previous_state = FLAMEGPU->getVariable<int>("previous_state");
    int unemployed = FLAMEGPU->getVariable<int>("unemployed");
    int type_resource = FLAMEGPU->getVariable<int>("type_resource");
    int t_arrival = FLAMEGPU->getVariable<int>("t_arrival");
    int language_level = FLAMEGPU->getVariable<int>("language_level");
    double education_level = FLAMEGPU->getVariable<double>("education_level");
    double Time_for_assimilation = FLAMEGPU->getVariable<double>("Time_for_assimilation");

    int max_natives_age = FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_natives");
    int max_migrants_age = FLAMEGPU->environment.getProperty<uint8_t>("Average_life_time_of_migrants");
    int min_natives_age = FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_natives");
    int min_migrants_age = FLAMEGPU->environment.getProperty<uint8_t>("Age_for_married_and_kids_birth_of_migrants");
    int min_natives_comfort = FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_natives");
    int min_migrants_comfort = FLAMEGPU->environment.getProperty<uint8_t>("Minimum_comfort_level_of_migrants");
    int pension_age = FLAMEGPU->environment.getProperty<uint8_t>("Pension_age");


    FLAMEGPU->setVariable<int>("kids_should_be_born_native", 0);  // По умолчанию новых детей нет
    FLAMEGPU->setVariable<int>("kids_should_be_born_migrant", 0);  

    if (type == 1)
        FLAMEGPU->setVariable<double>("Time_for_assimilation", 0.0);

    double t_assim = 30; // Максимальное врмя на ассимиляцию

    double koeff = FLAMEGPU->environment.getProperty<float>("Average_GDP_rate") * (FLAMEGPU->environment.getProperty<double>("Expenditure_on_education_share"));
    if (koeff <= 0)
        koeff = 1;

    education_level = 3 * exp(-1 / (double)(koeff * (1 + language_level))); // Уровень образования

    if (language_level > 0 && education_level > 0)
        t_assim = 30 * pow(1 / (double)(1 + language_level), 0.7) * pow(1 / (double)(1 + education_level), 0.3); //Вклад уровня образования

    if (t_assim > 30)
        t_assim = 30;
    if (t_assim < 0)
        t_assim = 0;

    if (type == 2)
        FLAMEGPU->setVariable<double>("Time_for_assimilation", t_assim);


    //Обновляем уровень личного комфорта агента
    if (comfort_level > 0 && comfort_level < 10 && type == 1) // Гражданина
    {
        if (type_resource == 2) //Высокотехнологичный ресурс
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level + 1);
        if (type_resource == 1) //Низкотехнологичный ресурс
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level + 0.5);
        if (type_resource == 0) // Безработный
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level - 1);
    }

    if (comfort_level > 0 && comfort_level < 10 && type == 2) // Мигранта
    {
        if (type_resource == 2) //Высокотехнологичный ресурс
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level + 1);
        if (type_resource == 1) //Низкотехнологичный ресурс
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level + 1);
        if (type_resource == 0) // Безработный
            FLAMEGPU->setVariable<double>("comfort_level", comfort_level - 1);
    }

    // Стационароное состояние агента
    if ((previous_state == 2 && comfort_level >= min_natives_comfort) ||
        ((previous_state == 3 || previous_state == 4 || previous_state == 5) && current_age > 50) && type == 1 ||

        (previous_state == 2 && comfort_level >= min_migrants_comfort) ||
        ((previous_state == 3 || previous_state == 4 || previous_state == 5) && current_age > 50) && type == 2)
        FLAMEGPU->setVariable<int>("current_state", 1);  // Стационарное   


 // Поиск работы
    if ((comfort_level < min_natives_comfort &&
        current_age >= 18 &&
        current_age <= pension_age && unemployed == 1 && type == 1 && (previous_state == 1 || previous_state == 4 || previous_state == 5)) ||

        (comfort_level < min_migrants_comfort &&
            current_age >= 18 &&
            current_age <= pension_age && unemployed == 1 && type == 2 && (previous_state == 1 || previous_state == 4 || previous_state == 5)))
        FLAMEGPU->setVariable<int>("current_state", 2);  // Поиск работы   



// Поиск партнера для вступления в брак
    if ((comfort_level >= min_natives_comfort &&
        current_age >= min_natives_age &&
        current_age <= 50 && married == 0 && previous_state == 1 && type == 1) ||

        (comfort_level >= min_migrants_comfort &&
            current_age >= min_migrants_age &&
            current_age <= 50 && married == 0 && previous_state == 1 && type == 2))
        FLAMEGPU->setVariable<int>("current_state", 3); // Поиск партнера для брака и рождения детей     


   //Готовность к рождению детей
    if ((comfort_level >= min_natives_comfort &&
        current_age >= min_natives_age &&
        current_age <= 50 && married == 1 && type == 1 && (previous_state == 3 || previous_state == 1)) ||

        (comfort_level >= min_migrants_comfort &&
            current_age >= min_migrants_age &&
            current_age <= 50 && married == 1 && type == 2 && (previous_state == 3 || previous_state == 1)))
        FLAMEGPU->setVariable<int>("current_state", 4); // Готовоность к рождению детей    


    double p = FLAMEGPU->random.uniform<double>();
	double g = -1 / (pow((double)kids, 0.9) * pow((double)current_age, 0.1));
	double w = (double)exp(g);
    
    if ((previous_state == 4 && p < 1 * (1 - w)) ||
        (previous_state == 1 && married == 0 && p < 0.01 * (1 - w)))
    {
        FLAMEGPU->setVariable<int>("current_state", 5); // Рождение ребёнка   
        if (type == 1)
            FLAMEGPU->setVariable<int>("kids_should_be_born_native", 1);
        if (type == 2)
            FLAMEGPU->setVariable<int>("kids_should_be_born_migrant", 1);


                
        FLAMEGPU->setVariable<int>("kids", FLAMEGPU->getVariable<int>("kids") + 1); // Рождение ребёнка   
    }


    //Ассимиляция мигрантов
    if (FLAMEGPU->getStepCounter() - t_arrival > t_assim && type == 2)
    {
        //  printf("%f, %i, %f\n", t_assim, language_level, education_level);

          //Добавление агента-мигранта в новую популяцию агентов-граждан
        FLAMEGPU->setVariable<int>("current_state", 6);
        //Добавление агента-мигранта в новую популяцию агентов-граждан
        FLAMEGPU->setVariable<int>("type", 1);
        FLAMEGPU->setVariable<int>("is_assimilated", 1); // Метка о том, что это бывший ассимилированный мигрант
    }


    FLAMEGPU->setVariable<int>("previous_state", FLAMEGPU->getVariable<int>("current_state"));

    if (current_age >= FLAMEGPU->environment.getProperty<uint8_t>("Pension_age"))
        FLAMEGPU->setVariable<int>("is_pensioner", 1);



    //Выбытие агентов по естественным причинам 
    if ((current_age <= max_natives_age && FLAMEGPU->getVariable<int>("type") == 1) ||
        (current_age <= max_migrants_age && FLAMEGPU->getVariable<int>("type") == 2))
        return flamegpu::ALIVE;
    else
        return flamegpu::DEAD;
}


int main(int argc, const char** argv) {

    flamegpu::ModelDescription model("Migration");

    flamegpu::MessageArray2D::Description& message_resource = model.newMessage<flamegpu::MessageArray2D>("location_resource");
    {
        message_resource.newVariable<int>("id");
        message_resource.newVariable<float>("x");
        message_resource.newVariable<float>("y");

        message_resource.newVariable<int>("type_resource"); // Передача типа ячеёки для обновления состояния ресурса
        message_resource.newVariable<int>("is_occupied"); // Передача состояния ячеёки
        message_resource.newVariable<int>("row"); // Координаты ячейки в дискретной системе координат
        message_resource.newVariable<int>("clm");
        message_resource.newVariable<int>("request_agent");

        message_resource.setDimensions(dim, dim);
    }


    flamegpu::MessageArray2D::Description& message_agent = model.newMessage<flamegpu::MessageArray2D>("location_agent");
    {
        message_agent.newVariable<float>("x");
        message_agent.newVariable<float>("y");

        message_agent.newVariable<int>("id");
        message_agent.newVariable<int>("row"); // Координаты агента в дискретной системе координат
        message_agent.newVariable<int>("clm");

        message_agent.newVariable<int>("move");


        message_agent.newVariable<int>("current_state");
        message_agent.newVariable<int>("type");
        message_agent.newVariable<int>("married");
        message_agent.newVariable<int>("gender");

        message_agent.newVariable<double>("education_level");

        message_agent.setDimensions(dim, dim);
    }


    flamegpu::MessageArray2D::Description& message_married = model.newMessage<flamegpu::MessageArray2D>("agents_married");
    {
        message_married.newVariable<int>("ID_partner");
        message_married.newVariable<int>("ID_married");

        message_married.setDimensions(dim, dim);
    }


    flamegpu::MessageArray2D::Description& message_occupied = model.newMessage<flamegpu::MessageArray2D>("resource_occupied");
    {

        message_occupied.newVariable<int>("id");// Текущие координаты ресурса
        message_occupied.newVariable<float>("x");
        message_occupied.newVariable<float>("y");
        message_occupied.newVariable<int>("row");
        message_occupied.newVariable<int>("clm");

        message_occupied.newVariable<float>("target_x");// Координаты целевого ресурса
        message_occupied.newVariable<float>("target_y");
        message_occupied.newVariable<int>("target_row");
        message_occupied.newVariable<int>("target_clm");

        message_occupied.newVariable<int>("agent_ID"); // 
        message_occupied.newVariable<int>("target_ID"); // 

        message_occupied.setDimensions(dim, dim);
    }

    flamegpu::AgentDescription& agent = model.newAgent("agent");
    {
        agent.newVariable<int>("id"); // ID
        agent.newVariable<float>("x"); // Координаты агента в непрервной системе координат
        agent.newVariable<float>("y");
        agent.newVariable<int>("type"); // 1 - Коренные жители, 2 - мигранты 
        agent.newVariable<int>("row"); // Координаты агента в дискретной системе координат
        agent.newVariable<int>("clm");
        agent.newVariable <unsigned int, 2>("pos"); // Позиция агента в двумерном пространстве
        agent.newVariable<int>("current_state");
        agent.newVariable<int>("previous_state");
        agent.newVariable<int>("unemployed");
        agent.newVariable<int>("t_arrival");
        agent.newVariable<int>("age");
        agent.newVariable<int>("gender");
        agent.newVariable<double>("education_level");
        agent.newVariable<int>("language_level");
        agent.newVariable<double>("comfort_level");
        agent.newVariable<int>("married");
        agent.newVariable<int>("kids");
        agent.newVariable<int>("kids_should_be_born_native");
        agent.newVariable<int>("kids_should_be_born_migrant");
        agent.newVariable<double>("Time_for_assimilation");
        agent.newVariable<int>("ID_partner");
        agent.newVariable<int>("type_resource"); // Тип ресурса - рабочего места, который занимает агент
        agent.newVariable<int>("move"); // Признак того, что агент должен перемещаться в произвольную ячеёку дискретного пространства
      //  agent.newVariable<int>("CI");
        agent.newVariable<int>("is_pensioner"); // Является ли пенсионером
        agent.newVariable<int>("is_assimilated"); // Является ли ассимилированным мигрантом
    }



    auto& fn_all_agents = agent.newFunction("all_agents", all_agents);
    {
        fn_all_agents.setMessageOutput("location_agent"); // Информация по всем агентам
    }

   
    auto& fn_check_all_agents = agent.newFunction("check_all_agents", check_all_agents);
    {
        fn_check_all_agents.setMessageInput("location_agent"); 
        fn_check_all_agents.setAllowAgentDeath(true); // Удаление агента случайно оказавшегося в той же ячейке
    }


    auto& fn_neighbour_agents1 = agent.newFunction("agent_to_agent_contacts", agent_to_agent_contacts);
    {
        fn_neighbour_agents1.setMessageInput("location_agent");
    }


    auto& fn_agent_location_update = agent.newFunction("check_cell", check_cell);
    {
        fn_agent_location_update.setMessageInput("location_resource");
       // fn_agent_location_update.setMessageOutput("location_agent");
    }


    auto& fn_state_update = agent.newFunction("update_agent_state", update_agent_state);  // Обновление состояния агента

    {
        fn_state_update.setAllowAgentDeath(true);
    }

    auto& fn_looking_for_partner = agent.newFunction("looking_for_partner", looking_for_partner); // Обработка данных, поступающих от потенциальных партнеров
    {
        fn_looking_for_partner.setMessageInput("location_agent");
        fn_looking_for_partner.setMessageOutput("agents_married");
    }


    auto& fn_married = agent.newFunction("get_married", get_married); // Отправка данных от агентов о своем местоположении и статусе
    {
        fn_married.setMessageInput("agents_married");
    }

    auto& fn_moving_trasaction = agent.newFunction("moving_trasaction", moving_trasaction); // Обработка данных, поступающих от потенциальных партнеров
    {
        fn_moving_trasaction.setMessageInput("resource_occupied");
    }

    flamegpu::AgentDescription& resources = model.newAgent("resources");
    {
        resources.newVariable<int>("id"); // ID
        resources.newVariable<float>("x"); // Координаты агента в непрервной системе координат
        resources.newVariable<float>("y");
        resources.newVariable<int>("type_resource"); // 2 - Высокотехнологичные, 1 - низкотехнологичные, 0 - безработные
        resources.newVariable<int>("row"); // Координаты агента в дискретной системе координат
        resources.newVariable<int>("clm");
        resources.newVariable <unsigned int, 2>("pos"); // Позиция ресурса в двумерном пространстве
        resources.newVariable<int>("time_creation"); // Момент времени создания ресурса
        resources.newVariable<int>("is_occupied"); // Ресурс занят
        resources.newVariable<int>("waiting_occupation"); // Ресурс ожидает занятие агентом
        resources.newVariable<int>("request_agent"); // ID агента, который может переместиться в данную ячейку
        //resources.newVariable<int>("CI"); // индекс кластеризации агента (по умолчанию - 1)
    }


    auto& fn_all_resources = resources.newFunction("all_resources", all_resources);
    {
        fn_all_resources.setMessageOutput("location_resource"); // Информация по всем ресурсам
    }


    auto& fn_resources_location_update = resources.newFunction("update_cell", update_cell);
    {
        fn_resources_location_update.setMessageInput("location_agent");
       // fn_resources_location_update.setMessageOutput("location_resource");
    }

    auto& fn_workplaces_creation = resources.newFunction("workplaces_creation", workplaces_creation); // Создание новых рабочих мес
    {
        fn_workplaces_creation.setMessageInput("location_resource");
    }


    auto& fn_looking_for_resource = resources.newFunction("looking_for_resource", looking_for_resource); // "Ресурс ищет агента"
    {
        fn_looking_for_resource.setMessageInput("location_agent");
        fn_looking_for_resource.setMessageOutput("resource_occupied"); // ID ресурса, который должен занять определенный агент
    }


    /**
    * Control flow
    */

    model.addInitFunction(init_function);
    model.addStepFunction(cells_update);
    model.addStepFunction(BasicOutput);
    model.addExitCondition(exit_condition);
       

    { // Layer  #0 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(all_agents);
    }

    { // Layer  #1 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(all_resources);
    }

    

    {   // Layer #2 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(check_all_agents);
    }

    

    {   // Layer #3 Функции агента
       flamegpu::LayerDescription& layer = model.newLayer();
       layer.addAgentFunction(check_cell);
    }

    
    {   // Layer #4 Функции ресурса
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(update_cell);
    }

  
    
    {   // Layer #5 Функции ресурса
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(workplaces_creation);
    }

    
    {   // Layer #6 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(update_agent_state);
    }


    {   // Layer #7 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(agent_to_agent_contacts);
    }


    {   // Layer #8 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(looking_for_partner);
    }


    {   // Layer #9 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(get_married);
    }


    {   // Layer #10 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(looking_for_resource);
    }


    {   // Layer #11 Функции агента
        flamegpu::LayerDescription& layer = model.newLayer();
        layer.addAgentFunction(moving_trasaction);
    }
    
    {
        flamegpu::EnvironmentDescription& env = model.Environment();
        //Управляющие параметры Модели
        env.newProperty<uint8_t>("Life_age_high_technology_work_places", 15);
        env.newProperty<uint8_t>("Life_age_low_technology_work_places", 15);
        env.newProperty<uint8_t>("Method_work_places_creation", 1);
        env.newProperty<uint8_t>("Frequency_work_places_creation", 5);
        env.newProperty<uint8_t>("Average_life_time_of_natives", 80);
        env.newProperty<uint8_t>("Average_life_time_of_migrants", 65);

        env.newProperty<double>("Share_of_new_migrants", 0.1);
        env.newProperty<double>("Expenditure_on_education_share", 0.1); // Доля затрат на образования в ВВП на душу населения


        env.newProperty<uint8_t>("Age_for_married_and_kids_birth_of_natives", 29);
        env.newProperty<uint8_t>("Age_for_married_and_kids_birth_of_migrants", 18);
        env.newProperty<uint8_t>("Minimum_comfort_level_of_natives", 6);
        env.newProperty<uint8_t>("Minimum_comfort_level_of_migrants", 3);
        env.newProperty<uint8_t>("Pension_age", 65);

        env.newProperty<float>("Influence_of_education_on_employment_opportunities", 0.5);

        //Макроэкономические показатели
        env.newProperty<float>("GE", 0);
        env.newProperty<float>("GDP", 0);

        env.newProperty<float>("GE_rate_total", 0);
        env.newProperty<float>("GDP_rate_total", 0);
        
        env.newProperty<float>("Average_Government_Expenditure_rate", 0);
        env.newProperty<float>("Average_GDP_rate", 0);

        env.newProperty<float>("Total_DSI", 0);
        env.newProperty<float>("Average_DSI", 0);

        env.newProperty<int, 2>("Center_of_first_high_tech_cluster", { 0, 0 }); // Координаты центров кластеров для создания рабочих мест
        env.newProperty<int, 2>("Center_of_second_high_tech_cluster", { 0, 0 });
        env.newProperty<int, 2>("Center_of_third_high_tech_cluster", { 0, 0 });

        env.newProperty<int, 2>("Center_of_first_low_tech_cluster", { 0, 0 });
        env.newProperty<int, 2>("Center_of_second_low_tech_cluster", { 0, 0 });
        env.newProperty<int, 2>("Center_of_third_low_tech_cluster", { 0, 0 });

        env.newProperty<int>("Size_of_high_tech_cluster", 10);
        env.newProperty<int>("Size_of_low_tech_cluster", 10);
        
    }

    /**
     * Create Model Runner
     */

    flamegpu::CUDASimulation cuda_model(model);
    cuda_model.initialise(1, argv);
    cuda_model.SimulationConfig().steps = TIME_STOP;
    flamegpu::AgentVector population1(model.Agent("agent"), NumberOfCitizens + NumberOfMigrants); 
    flamegpu::AgentVector population2(model.Agent("resources"), dim*dim);



    //Множественный запуск без визуализации
    if(VIS_MODE==0)
    { 
        flamegpu::RunPlanVector runs(model, RUN_COUNT); // 2 - количество вариаций
        {
            runs.setSteps(TIME_STOP);

            runs.setRandomSimulationSeed(12, 1);

            runs.setOutputSubdirectory("results");

            runs.setPropertyUniformRandom<double>("Share_of_new_migrants", double(0.1f), double(0.5f));
            runs.setPropertyUniformRandom<double>("Expenditure_on_education_share", double(0.1f), double(0.5f));

            runs.setPropertyUniformDistribution<uint8_t>("Life_age_high_technology_work_places", 5, 15);
            runs.setPropertyUniformDistribution<uint8_t>("Life_age_low_technology_work_places", 5, 15);
            runs.setPropertyUniformDistribution<uint8_t>("Method_work_places_creation", 1, 2);
            runs.setPropertyUniformDistribution<uint8_t>("Frequency_work_places_creation", 5, 5);
            runs.setPropertyUniformDistribution<uint8_t>("Average_life_time_of_natives", 70, 90);
            runs.setPropertyUniformDistribution<uint8_t>("Average_life_time_of_migrants", 60, 80);

            runs.setPropertyUniformDistribution<uint8_t>("Age_for_married_and_kids_birth_of_natives", 20, 35);
            runs.setPropertyUniformDistribution<uint8_t>("Age_for_married_and_kids_birth_of_migrants", 18, 30);
            runs.setPropertyUniformDistribution<uint8_t>("Minimum_comfort_level_of_natives", 6, 10);
            runs.setPropertyUniformDistribution<uint8_t>("Minimum_comfort_level_of_migrants", 3, 6);
            runs.setPropertyUniformDistribution<uint8_t>("Pension_age", 60, 75);

        }

        flamegpu::CUDAEnsemble cuda_ensemble(model, 1, argv);
        cuda_ensemble.simulate(runs);
    }
 
    //Одиночный запуск с визуализацией
    if (VIS_MODE == 1)
    {
        initVisualisation();
        glutTimerFunc(1, timer, 0);
        cuda_model.SimulationConfig().steps = TIME_STOP;
        std::thread first([&cuda_model]() { cuda_model.simulate(); });
        runVisualisation(); //Запуск визуализации
        first.join();
    }
    
     

    // Выгрузка полученных рехультатов (только для ансамбля)

   for (int i = 0; i < RUN_COUNT; i++)
   {
       if (out.is_open())
	   {
           if (i == 0)
           {
               out << "Share_of_new_migrants" <<
                   ";" << "Expenditure_on_education_share" <<
                   ";" << "Life_age_high_technology_work_places" <<
                   ";" << "Life_age_low_technology_work_places" <<
                   ";" << "Frequency_work_places_creation" <<
                   ";" << "Average_life_time_of_natives" <<
                   ";" << "Average_life_time_of_migrants" <<
                   ";" << "Age_for_married_and_kids_birth_of_natives" <<
                   ";" << "Age_for_married_and_kids_birth_of_migrants" <<
                   ";" << "Minimum_comfort_level_of_natives" <<
                   ";" << "Minimum_comfort_level_of_migrants" <<
                   ";" << "Pension_age" <<
                   ";" << "Method_work_places_creation" <<

                   ";" << "Total_count_of_agents" <<
                   ";" << "Share_of_non-assimilated_migrants" <<
                   ";" << "Number_of_assimilated_migrants" <<
                   ";" << "Averaged_time_for_assimilation" <<
                   ";" << "Duncan_Segregation_Index" <<
                   ";" << "Average_GDP_rate" <<
                   ";" << "Average_Government_Expenditure_rate" << std::endl;
           }



		   //Выгрузка полученных результатов
           
           float par1 = (float)myArray1[i].load();
           float par2 = (float)myArray2[i].load();
           float par3 = (float)myArray15[i].load();
           float par4 = (float)myArray17[i].load();
           float par5 = (float)myArray18[i].load();
           float par6 = (float)myArray19[i].load();
           float par7 = (float)myArray20[i].load();

             out << par1 / 100000 <<
               ";" << par2 / 100000 <<
               ";" << myArray3[i].load() <<
               ";" << myArray4[i].load() <<
               ";" << myArray5[i].load() <<
               ";" << myArray6[i].load() <<
               ";" << myArray7[i].load() <<
               ";" << myArray8[i].load() <<
               ";" << myArray9[i].load() <<
               ";" << myArray10[i].load() <<
               ";" << myArray11[i].load() <<
               ";" << myArray12[i].load() <<
               ";" << myArray13[i].load() <<

               ";" << myArray14[i].load() <<
               ";" << par3 / 100000 <<
			   ";" << myArray16[i].load() <<
			   ";" << par4 / 100000 <<
			   ";" << par5 / 100000 <<
               ";" << par6 / 100000 <<
			   ";" << par7 / 100000 << std::endl;
	   }

   }

    out.close();


    //Вывод логов отражающих работу модели на уровне потоков (для ансамбля)
   
    for (int i = 0; i < RUN_COUNT; i++)
    { 
        if (out2.is_open())
        {
            //Выгрузка полученных результатов
            out2 << myArray_id[i] <<
             ";" << myArray_test1[i] <<
             ";" << myArray_test2[i] << 
             ";" << myArray_test3[i] <<  std::endl;
        }
    }

    out2.close();
    

    return 0;
}
