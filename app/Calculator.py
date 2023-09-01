import sympy as smp
from sympy import *
from sympy.vector import *
from sympy.plotting import plot3d
from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt


x, y, z, t = smp.symbols("x y z t")
C = CoordSys3D('')

class InvalidOptionError(Exception):
    """
    This exception is raised whenever the user enters an input that is 
    not one of the options availiable
    """
    pass

class CalculatorApp:

    OK = 0
    ERROR = 1

    def __init__(self, f):
        self.f = f
    
    def evaluate_at_1D_point(self, point) -> smp.latex:
        """
        Displays the value a single-variable function evaluated at the given point
        Returns 0 on success and 1 otherwise
        """
        try:
            smp.sympify(point)
        except SympifyError:
            print("Invalid point, please try again!")
            return self.ERROR
        
        sub_function = self.f.subs({'x': UnevaluatedExpr(point)})
        evaluated_function = self.f.subs({'x':point}).evalf()
        display(Math(smp.latex(Eq(sub_function, evaluated_function, evaluate = False))))
        return smp.latex(Eq(sub_function, evaluated_function, evaluate = False))
    
    def evaluate_at_2D_or_3D_point(self, ordered_pair) -> int:
        """
        Displays the value of the 2 or 3 variable function at the given point
        Returns 0 on success and 1 otherwise
        """
        point_list = [i for i in ordered_pair]
        if len(point_list) == 2:
            try:
                x_point = smp.sympify(point_list[0])
                y_point = smp.sympify(point_list[1])
            except SympifyError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            
            inputted_function = self.f.subs({'x': UnevaluatedExpr(x_point), 'y': UnevaluatedExpr(y_point)})
            evaluated_function = self.f.subs({'x':x_point,'y':y_point}).evalf()
            display(Math(smp.latex(Eq(inputted_function, evaluated_function, evaluate = False))))
            return smp.latex(Eq(inputted_function, evaluated_function, evaluate = False))
        
        elif len(point_list) == 3:
            try:
                x_point = smp.sympify(point_list[0])
                y_point = smp.sympify(point_list[1])
                z_point = smp.sympify(point_list[2])
            except SympifyError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            
            inputted_function = self.f.subs({'x': UnevaluatedExpr(x_point),'y': UnevaluatedExpr(y_point),'z': UnevaluatedExpr(z_point)})
            evaluated_function = self.f.subs({'x':x_point,'y':y_point,'z':z_point}).evalf()
            display(Math(smp.latex(Eq(inputted_function, evaluated_function, evaluate = False))))
            return smp.latex(Eq(inputted_function, evaluated_function, evaluate = False))
        else:
            print("Too many inputs in the ordered pair! Please try again.")
            return self.ERROR
        
    def graph_2D(self, x_interval) -> int:
        """
        Produces a 2-D graph of a single-variable function over the given interval
        Returns 0 on success and 1 otherwise
        """
        x_list = [i for i in x_interval]
        try:
            lower_x = smp.sympify(x_list[0])
            upper_x = smp.sympify(x_list[1])
        except SympifyError:
            print("Invalid input! Please try again.")
            return self.ERROR
        
        func = smp.lambdify(x, self.f)
        x_plot = np.linspace(int(lower_x), int(upper_x), 1000)
        y_plot = func(x_plot)
        fig, ax = plt.subplots()
        ax.plot(x_plot, y_plot)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

        return fig
    
    def graph_3D(self, x_interval, y_interval) -> int:
        """
        Produces a 3-D graph of a 2 variable function over the given intervals
        Returns 0 on success and 1 otherwise
        """
        x_list = [i for i in x_interval]
        y_list = [i for i in y_interval]
        try:
            lower_x = smp.sympify(x_list[0])
            upper_x = smp.sympify(x_list[1])
            lower_y = smp.sympify(y_list[0])
            upper_y = smp.sympify(y_list[1])                
        except SympifyError:
            print("Please enter a valid number!")
            return self.ERROR
        
        func = smp.lambdify([x, y], self.f)
        x_plot = np.linspace(int(lower_x), int(upper_x), 1000)
        y_plot = np.linspace(int(lower_y), int(upper_y), 1000)
        X, Y = np.meshgrid(x_plot, y_plot)
        z_plot = func(X, Y)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, z_plot, cmap=plt.cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return fig

    def differentiate(self, variable) -> int:
        """
        Displays the derivative of a function with respect to the given variable
        Returns 0 on success and 1 otherwise
        """
        if variable != "x" and variable != "y" and variable != "z":                    
            print("Not a valid variable! Please try again.") 
            return self.ERROR
        
        if variable == "x":
            partial_x_expression = smp.Derivative(self.f,x)
            partial_x_evaluated = smp.diff(self.f,x)
            display(Math(smp.latex(Eq(partial_x_expression, partial_x_evaluated, evaluate = False))))
            return smp.latex(Eq(partial_x_expression, partial_x_evaluated, evaluate = False))
        elif variable == "y":
            partial_y_expression = smp.Derivative(self.f,y)
            partial_y_evaluated = smp.diff(self.f,y)
            display(Math(smp.latex(Eq(partial_y_expression, partial_y_evaluated, evaluate = False))))
            return smp.latex(Eq(partial_y_expression, partial_y_evaluated, evaluate = False))
        elif variable == "z":
            partial_z_expression = smp.Derivative(self.f,z)
            partial_z_evaluated = smp.diff(self.f,z)
            display(Math(smp.latex(Eq(partial_z_expression, partial_z_evaluated, evaluate = False))))
            return smp.latex(Eq(partial_z_expression, partial_z_evaluated, evaluate = False))
        return self.OK
    
    def integrate(self):
        """
        Displays the indefinite integral (antiderivative) of a function
        """
        indefinite_integral = smp.Integral(self.f,x)
        result_indefinite = smp.integrate(self.f,x)
        display(Math(smp.latex(Eq(indefinite_integral, result_indefinite, evaluate = False))))
        return smp.latex(Eq(indefinite_integral, result_indefinite, evaluate = False))
        
    def integrate_1D(self, x_interval) -> int:
        """
        Displays the value of the definite integral of a single-variable function over the given interval
        Returns 0 on success and 1 otherwise
        """
        x_range = [i for i in x_interval]
        try:
            lower_x = smp.sympify(x_range[0])
            upper_x = smp.sympify(x_range[1])
        except SympifyError:
            print("Not a valid interval, please try again!")
            return self.ERROR
        
        integral_expression = smp.Integral(self.f, (x, lower_x, upper_x))
        integral_value = smp.integrate(self.f, (x, lower_x, upper_x)).evalf()
        display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
        return smp.latex(Eq(integral_expression, integral_value, evaluate = False))

    def integrate_2D(self, x_interval, y_interval, order) -> int:
        """
        Displays the value of the double integral of a 2 variable function over the given intervals
        Returns 0 on success and 1 otherwise
        """
        x_range = [i for i in x_interval]
        y_range = [i for i in y_interval]
        try:
            lower_x = smp.sympify(x_range[0])
            upper_x = smp.sympify(x_range[1])
            lower_y = smp.sympify(y_range[0])
            upper_y = smp.sympify(y_range[1])                         
        except SympifyError:
            print("Please enter a valid expression!")
            return self.ERROR
        
        if order == "dxdy":
            integral_expression = smp.Integral(self.f, (x,lower_x,upper_x),(y,lower_y,upper_y))
            integral_value = smp.integrate(self.f, (x,lower_x,upper_x),(y,lower_y,upper_y)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dydx":
            integral_expression = smp.Integral(self.f, (y,lower_y,upper_y),(x,lower_x,upper_x))
            integral_value = smp.integrate(self.f, (y,lower_y,upper_y),(x,lower_x,upper_x)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
        return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
    
    def integrate_3D(self, x_interval, y_interval, z_interval, order) -> int:
        """
        Displays the value of the triple integral of f(x,y,z) over the given intervals
        Returns 0 on success and 1 otherwise
        """
        x_range = [i for i in x_interval]
        y_range = [i for i in y_interval]
        z_range = [i for i in z_interval]
        try:
            lower_x = smp.sympify(x_range[0])
            upper_x = smp.sympify(x_range[1])
            lower_y = smp.sympify(y_range[0])
            upper_y = smp.sympify(y_range[1])
            lower_z = smp.sympify(z_range[0])
            upper_z = smp.sympify(z_range[1])
        except SympifyError:
            print("Please enter a valid expression!")
            return self.ERROR
        
        if order == "dxdydz":
            integral_expression = smp.Integral(self.f, (x,lower_x,upper_x),(y,lower_y,upper_y), (z,lower_z,upper_z))
            integral_value = smp.integrate(self.f, (x,lower_x,upper_x),(y,lower_y,upper_y), (z,lower_z,upper_z)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dxdzdy":
            integral_expression = smp.Integral(self.f, (x,lower_x,upper_x),(z,lower_z,upper_z), (y,lower_y,upper_y))
            integral_value = smp.integrate(self.f, (x,lower_x,upper_x),(z,lower_z,upper_z), (y,lower_y,upper_y)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dydxdz":
            integral_expression = smp.Integral(self.f, (y,lower_y,upper_y),(x,lower_x,upper_x), (z,lower_z,upper_z))
            integral_value = smp.integrate(self.f, (y,lower_y,upper_y),(x,lower_x,upper_x), (z,lower_z,upper_z))
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False)))) 
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dydzdx":
            integral_expression = smp.Integral(self.f, (y,lower_y,upper_y),(z,lower_z,upper_z), (x,lower_x,upper_x))
            integral_value = smp.integrate(self.f, (y,lower_y,upper_y),(z,lower_z,upper_z), (x,lower_x,upper_x)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dzdxdy":
            integral_expression = smp.Integral(self.f, (z,lower_z,upper_z),(x,lower_x,upper_x), (y,lower_y,upper_y))
            integral_value = smp.integrate(self.f, (z,lower_z,upper_z),(x,lower_x,upper_x), (y,lower_y,upper_y)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
        elif order == "dzdydx":
            integral_expression = smp.Integral(self.f, (z,lower_z,upper_z),(y,lower_y,upper_y), (x,lower_x,upper_x))
            integral_value = smp.integrate(self.f, (z,lower_z,upper_z),(y,lower_y,upper_y), (x,lower_x,upper_x)) 
            display(Math(smp.latex(Eq(integral_expression, integral_value, evaluate = False))))
            return smp.latex(Eq(integral_expression, integral_value, evaluate = False))
    
    def directional_derivative(self, point, vector) -> int: 
        """
        Displays the value of the directional derivative of the function at the given point and direction
        Returns 0 on success and 1 otherwise
        """
        point_list = [i for i in point]
        if len(point_list) == 2:
            try:
                x_coord = smp.sympify(point_list[0])
                y_coord = smp.sympify(point_list[1])    
            except SympifyError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            
            try:
                vector_components = [i for i in vector]
                vector_x = smp.sympify(vector_components[0])
                vector_y = smp.sympify(vector_components[1])                         
            except SympifyError:
                print("Invalid vector! Please try again.")
                return self.ERROR
            
            grad = diff(self.f,x).subs({'x':x_coord, 'y':y_coord})*C.i + diff(self.f,y).subs({'x':x_coord, 'y':y_coord})*C.j
            unit_vector = (vector_x*C.i + vector_y*C.j).normalize()
            directional_derivative = grad.dot(unit_vector)
            display(Math(smp.latex(directional_derivative)))
            return smp.latex(directional_derivative)
        
        elif len(point_list) == 3:
            try:
                x_coord = smp.sympify(point_list[0])
                y_coord = smp.sympify(point_list[1])
                z_coord = smp.sympify(point_list[2])            
            except SympifyError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            except IndexError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            
            try:
                vector_components = [i for i in vector]
                vector_x = smp.sympify(vector_components[0])
                vector_y = smp.sympify(vector_components[1])
                vector_z = smp.sympify(vector_components[2])            
            except SympifyError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            except IndexError:
                print("Invalid ordered pair! Please try again.")
                return self.ERROR
            
            grad = diff(self.f,x).subs({'x':x_coord, 'y':y_coord, 'z':z_coord})*C.i + diff(self.f,y).subs({'x':x_coord, 'y':y_coord, 'z':z_coord})*C.j + diff(self.f,z).subs({'x':x_coord, 'y':y_coord, 'z':z_coord})*C.k
            unit_vector = (vector_x*C.i + vector_y*C.j + vector_z*C.k).normalize()
            directional_derivative = grad.dot(unit_vector)
            display(Math(smp.latex(directional_derivative)))
            return smp.latex(directional_derivative)
        else:
            print("Too many inputs in ordered pair, please try again")
            return self.ERROR

    def gradient_2D(self):
        """
        Displays the gradient of a 2 variable function
        """
        grad = diff(self.f,x)*C.i + diff(self.f,y)*C.j
        display(Math(smp.latex(grad)))
        return smp.latex(grad)

    def gradient_3D(self):
        """
        Displays the gradient of a 3 variable function
        """
        grad = diff(self.f,x)*C.i + diff(self.f,y)*C.j + diff(self.f,z)*C.k
        display(Math(smp.latex(grad)))
        return smp.latex(grad)

    def line_integral_2D(self, x, y, t_interval) -> int:
        """
        Displays the value of the 2D scalar line integral of the function over the given parameterized curve and interval
        Returns 0 on success and 1 otherwise
        """
        t_list = [i for i in t_interval]
        try:
            curve_x = smp.sympify(x)
            curve_y = smp.sympify(y)
            lower_t = smp.sympify(t_list[0])
            upper_t = smp.sympify(t_list[1])
        except SympifyError:
            print("Please enter a valid expression!")
            return self.ERROR
        
        r = smp.Matrix([curve_x, curve_y])
        integrand = self.f.subs({'x': curve_x, 'y': curve_y}) * smp.diff(r,t).norm()
        line_integral_expression = smp.Integral(integrand, (t,lower_t,upper_t))
        line_integral_value = smp.integrate(integrand, (t,lower_t,upper_t))
        display(Math(smp.latex(Eq(line_integral_expression, line_integral_value, evaluate = False))))
        return smp.latex(Eq(line_integral_expression, line_integral_value, evaluate = False))
    
    def line_integral_3D(self, x, y, z, t_interval) -> int:
        """
        Displays the value of the 3D scalar line integral of the function over the given parameterized curve and interval
        Returns 0 on success and 1 otherwise
        """
        t_list = [i for i in t_interval]
        try:
            curve_x = smp.sympify(x)
            curve_y = smp.sympify(y)
            curve_z = smp.sympify(z)
            lower_t = smp.sympify(t_list[0])
            upper_t = smp.sympify(t_list[1])                 
        except SympifyError:
            print("Please enter a valid expression!")
            return self.ERROR
        
        r = smp.Matrix([curve_x, curve_y, curve_z])
        integrand = self.f.subs({'x': curve_x, 'y': curve_y, 'z': curve_z}) * smp.diff(r,t).norm()
        line_integral_expression = smp.Integral(integrand, (t,lower_t,upper_t))
        line_integral_value = smp.integrate(integrand, (t,lower_t,upper_t))
        display(Math(smp.latex(Eq(line_integral_expression, line_integral_value, evaluate = False))))
        return smp.latex(Eq(line_integral_expression, line_integral_value, evaluate = False))