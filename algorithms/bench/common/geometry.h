#pragma once
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "../parlay/parallel.h"
#include "../parlay/primitives.h"
using namespace std;

// *************************************************************
//    POINTS AND VECTORS (3d),  2d is below
// *************************************************************


  template <class Coord>
  class point3d;

  template <class Coord>
  class vector3d {
  public:
    using coord = Coord;
    using vector = vector3d;
    using point = point3d<coord>;
    coord x;
    coord y;
    coord z;
    vector3d(coord x, coord y, coord z) : x(x), y(y), z(z) {}
    vector3d() :x(0), y(0), z(0) {}
    vector3d(point p);
    vector3d(parlay::slice<coord*,coord*> p) : x(p[0]), y(p[1]), z(p[2]) {};
    vector operator+(vector op2) {
      return vector(x + op2.x, y + op2.y, z + op2.z);}
    vector operator-(vector op2) {
      return vector(x - op2.x, y - op2.y, z - op2.z);}
    point operator+(point op2);
    vector operator*(coord s) {return vector(x * s, y * s, z * s);}
    vector operator/(coord s) {return vector(x / s, y / s, z / s);}
    coord& operator[] (int i) {return (i==0) ? x : (i==1) ? y : z;}
    coord dot(vector v) {return x * v.x + y * v.y + z * v.z;}
    vector cross(vector v) {
      return vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
    }
    coord maxDim() {return max(x,max(y,z));}
    void print() {cout << std::setprecision(10) << ":(" << x << "," << y << "," << z << "):";}
    coord Length(void) { return sqrt(x*x+y*y+z*z);}
    coord sqLength(void) { return x*x+y*y+z*z;}
    static const int dim = 3;
  };

  template <class Coord>
  class point3d {
  public:
    using coord = Coord;
    using vector = vector3d<coord>;
    using point = point3d;
    coord x; coord y; coord z;
    int dimension() {return 3;}
    point3d(coord x, coord y, coord z) : x(x), y(y), z(z) {}
    point3d() : x(0), y(0), z(0) {}
    point3d(vector v) : x(v.x), y(v.y), z(v.z) {};
    point3d(parlay::slice<coord*,coord*> p) : x(p[0]), y(p[1]), z(p[2]) {};
    void print() {cout << ":(" << x << "," << y << "," << z << "):";}
    vector operator-(point op2) {
      return vector(x - op2.x, y - op2.y, z - op2.z);}
    point operator+(vector op2) {
      return point(x + op2.x, y + op2.y, z + op2.z);}
    point minCoords(point b) {
      return point(min(x,b.x),min(y,b.y),min(z,b.z)); }
    point maxCoords(point b) { 
      return point(max(x,b.x),max(y,b.y),max(z,b.z)); }
    coord& operator[] (int i) {return (i==0) ? x : (i==1) ? y : z;}
    int quadrant(point center) {
      int index = 0;
      if (x > center.x) index += 1;
      if (y > center.y) index += 2;
      if (z > center.z) index += 4;
      return index;
    }
    // returns a point offset by offset in one of 8 directions 
    // depending on dir (an integer from [0..7])
    point offsetPoint(int dir, coord offset) {
      coord xx = x + ((dir & 1) ? offset : -offset);
      coord yy = y + ((dir & 2) ? offset : -offset);
      coord zz = z + ((dir & 4) ? offset : -offset);
      return point(xx, yy, zz);
    }
    point changeCoords(std::vector<coord> v){
      return point(v[0], v[1], v[2]);
    }
    // checks if pt is outside of a box centered at this point with
    // radius hsize
    bool outOfBox(point pt, coord hsize) { 
      return ((x - hsize > pt.x) || (x + hsize < pt.x) ||
	      (y - hsize > pt.y) || (y + hsize < pt.y) ||
	      (z - hsize > pt.z) || (z + hsize < pt.z));
    }
    static const int dim = 3;
  };

  template <class coord>
  inline point3d<coord> vector3d<coord>::operator+(point3d<coord> op2) {
    return point3d<coord>(x + op2.x, y + op2.y, z + op2.z);}

  template <class coord>
  inline vector3d<coord>::vector3d(point3d<coord> p) { x = p.x; y = p.y; z = p.z;}

  // *************************************************************
  //    POINTS AND VECTORS (2d)
  // *************************************************************

  template <class Coord>
  class point2d;

  template <class Coord>
  class vector2d {
  public: 
    using coord = Coord;
    using point = point2d<coord>;
    using vector = vector2d;
    coord x; coord y;
    vector2d(coord x, coord y) : x(x), y(y) {}
    vector2d() : x(0), y(0)  {}
    vector2d(point p);
    vector2d(parlay::slice<coord*,coord*> p) : x(p[0]), y(p[1]) {};
    vector operator+(vector op2) {return vector(x + op2.x, y + op2.y);}
    vector operator-(vector op2) {return vector(x - op2.x, y - op2.y);}
    point operator+(point op2);
    vector operator*(coord s) {return vector(x * s, y * s);}
    vector operator/(coord s) {return vector(x / s, y / s);}
    coord operator[] (int i) {return (i==0) ? x : y;};
    coord dot(vector v) {return x * v.x + y * v.y;}
    coord cross(vector v) { return x*v.y - y*v.x; }  
    coord maxDim() {return max(x,y);}
    void print() {cout << ":(" << x << "," << y << "):";}
    coord Length(void) { return sqrt(x*x+y*y);}
    coord sqLength(void) { return x*x+y*y;}
    static const int dim = 2;
  };

  template <class coord>
  static std::ostream& operator<<(std::ostream& os, const vector3d<coord> v) {
    return os << v.x << " " << v.y << " " << v.z; }

  template <class coord>
  static std::ostream& operator<<(std::ostream& os, const point3d<coord> v) {
    return os << v.x << " " << v.y << " " << v.z;
  }

  template <class Coord>
  class point2d {
  public: 
    using coord = Coord;
    using vector = vector2d<coord>;
    using point = point2d;
    coord x; coord y; 
    int dimension() {return 2;}
    point2d(coord x, coord y) : x(x), y(y) {}
    point2d() : x(0), y(0) {}
    point2d(vector v) : x(v.x), y(v.y) {};
    point2d(parlay::slice<coord*,coord*> p) : x(p[0]), y(p[1]) {};
    void print() {cout << ":(" << x << "," << y << "):";}
    vector operator-(point op2) {return vector(x - op2.x, y - op2.y);}
    point operator+(vector op2) {return point(x + op2.x, y + op2.y);}
    coord operator[] (int i) {return (i==0) ? x : y;};
    point minCoords(point b) { return point(min(x,b.x),min(y,b.y)); }
    point maxCoords(point b) { return point(max(x,b.x),max(y,b.y)); }
    int quadrant(point center) {
      int index = 0;
      if (x > center.x) index += 1;
      if (y > center.y) index += 2;
      return index;
    }
    // returns a point offset by offset in one of 4 directions 
    // depending on dir (an integer from [0..3])
    point offsetPoint(int dir, coord offset) {
      coord xx = x + ((dir & 1) ? offset : -offset);
      coord yy = y + ((dir & 2) ? offset : -offset);
      return point(xx,yy);
    }
    bool outOfBox(point pt, coord hsize) { 
      return ((x - hsize > pt.x) || (x + hsize < pt.x) ||
	      (y - hsize > pt.y) || (y + hsize < pt.y));
    }
    static const int dim = 2;
  };

  template <class coord>
  inline point2d<coord> vector2d<coord>::operator+(point2d<coord> op2) {
    return point2d<coord>(x + op2.x, y + op2.y);}

  template <class coord>
  inline vector2d<coord>::vector2d(point2d<coord> p) { x = p.x; y = p.y;}

  template <class coord>
  static std::ostream& operator<<(std::ostream& os, const vector2d<coord> v) {
    return os << v.x << " " << v.y;}

  template <class coord>
  static std::ostream& operator<<(std::ostream& os, const point2d<coord> v) {
    return os << v.x << " " << v.y; }

  // *************************************************************
  //    GEOMETRY
  // *************************************************************

  // Returns twice the area of the oriented triangle (a, b, c)
  template <class coord>
  inline coord triArea(point2d<coord> a, point2d<coord> b, point2d<coord> c) {
    return (b-a).cross(c-a);
  }

  template <class coord>
  inline coord triAreaNormalized(point2d<coord> a, point2d<coord> b, point2d<coord> c) {
    return triArea(a,b,c)/((b-a).Length()*(c-a).Length());
  }

  // Returns TRUE if the points a, b, c are in a counterclockise order
  template <class coord>
  inline bool counterClockwise(point2d<coord> a, point2d<coord> b, point2d<coord> c) {
    return (b-a).cross(c-a) > 0.0;
  }

  template <class coord>
  inline vector3d<coord> onParabola(vector2d<coord> v) {
    return vector3d<coord>(v.x, v.y, v.x*v.x + v.y*v.y);}

  // Returns TRUE if the point d is inside the circle defined by the
  // points a, b, c. 
  // Projects a, b, c onto a parabola centered with d at the origin
  //   and does a plane side test (tet volume > 0 test)
  template <class coord>
  inline bool inCircle(point2d<coord> a, point2d<coord> b, 
		       point2d<coord> c, point2d<coord> d) {
    vector3d<coord> ad = onParabola(a-d);
    vector3d<coord> bd = onParabola(b-d);
    vector3d<coord> cd = onParabola(c-d);
    return (ad.cross(bd)).dot(cd) > 0.0;
  }

  // returns a number between -1 and 1, such that -1 is out at infinity,
  // positive numbers are on the inside, and 0 is at the boundary
  template <class coord>
  inline double inCircleNormalized(point2d<coord> a, point2d<coord> b, 
				   point2d<coord> c, point2d<coord> d) {
    vector3d<coord> ad = onParabola(a-d);
    vector3d<coord> bd = onParabola(b-d);
    vector3d<coord> cd = onParabola(c-d);
    return (ad.cross(bd)).dot(cd)/(ad.Length()*bd.Length()*cd.Length());
  }

  // *************************************************************
  //    TRIANGLES
  // *************************************************************

  using tri = std::array<int,3>;

  template <class point>
  struct triangles {
    size_t numPoints() {return P.size();};
    size_t numTriangles() {return T.size();}
    parlay::sequence<point> P;
    parlay::sequence<tri> T;
    triangles() {}
    triangles(parlay::sequence<point> P, parlay::sequence<tri> T) 
      : P(std::move(P)), T(std::move(T)) {}
  };

  template <class point>
  struct ray {
    using vector = typename point::vector;
    point o;
    vector d;
    ray(point _o, vector _d) : o(_o), d(_d) {}
    ray() {}
  };

  template<class coord>
  inline coord angle(point2d<coord> a, point2d<coord> b, point2d<coord> c) {
    vector2d<coord> ba = (b-a);
    vector2d<coord> ca = (c-a);
    coord lba = ba.Length();
    coord lca = ca.Length();
    coord pi = 3.14159;
    return 180/pi*acos(ba.dot(ca)/(lba*lca));
  }

  template<class coord>
  inline coord minAngleCheck(point2d<coord> a, point2d<coord> b, point2d<coord> c, coord angle) {
    vector2d<coord> ba = (b-a);
    vector2d<coord> ca = (c-a);
    vector2d<coord> cb = (c-b);
    coord lba = ba.Length();
    coord lca = ca.Length();
    coord lcb = cb.Length();
    coord pi = 3.14159;
    coord co = cos(angle*pi/180.);
    return (ba.dot(ca)/(lba*lca) > co || ca.dot(cb)/(lca*lcb) > co || 
	    -ba.dot(cb)/(lba*lcb) > co);
  }

  template<class coord>
  inline point2d<coord> triangleCircumcenter(point2d<coord> a, point2d<coord> b, point2d<coord> c) {
    vector2d<coord> v1 = b-a;
    vector2d<coord> v2 = c-a;
    vector2d<coord> v11 = v1 * v2.dot(v2);
    vector2d<coord> v22 = v2 * v1.dot(v1);
    return a + vector2d<coord>(v22.y - v11.y, v11.x - v22.x)/(2.0 * v1.cross(v2));
  }

