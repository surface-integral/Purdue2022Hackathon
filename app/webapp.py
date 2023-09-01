import Calculator
import streamlit as st
import sympy as smp
from matplotlib.pyplot import plot

num_vars = st.radio("Select the number of variables in the function", [1, 2, 3])
input = st.text_input("Enter a function: ", "x")
input.replace("^", "**")
f = smp.sympify(input)
st.latex(smp.latex(f))
calculator = Calculator.CalculatorApp(f)

if num_vars == 1:
    option = st.radio("Select an action: ", ["Evaluate function at a point", "Graph function", 
                                    "Find the derivative of the function", "Indefinite Integral", 
                                    "Definite Integral"])
    if option == "Evaluate function at a point":
        point = st.number_input("Enter the point: ")
        st.latex(calculator.evaluate_at_1D_point(point))
    elif option == "Graph function":
        lower_x = st.number_input("Enter lower x bound: ")
        upper_x = st.number_input("Enter upper x bound: ")
        st.pyplot(calculator.graph_2D((lower_x, upper_x)))
    elif option == "Find the derivative of the function":
        st.latex(calculator.differentiate('x'))
    elif option == "Indefinite Integral":
        st.latex(calculator.integrate())
    elif option == "Definite Integral":
        lower = st.number_input("Enter the lower limit of integration: ")
        upper = st.number_input("Enter the upper limit of integration: ")
        st.latex(calculator.integrate_1D((lower, upper)))

elif num_vars == 2:
    option = st.radio("Select an action: ", ["Evaluate function at a point", "Graph function", 
                                                "Find a partial derivative of the function", 
                                                "Find gradient vector", "Find directional derivative", 
                                                "Evaluate the double integral of the function over a 2-D region", 
                                                "Evaluate the scalar line integral of the function in 2-space"])
    if option == "Evaluate function at a point":
        x_coord = st.number_input("Enter the x coordinate: ")
        y_coord = st.number_input("Enter the y coordinate: ")
        st.latex(calculator.evaluate_at_2D_or_3D_point((x_coord, y_coord)))
    elif option == "Graph function":
        x_lower = st.number_input("Enter the lower x bound: ")
        x_upper = st.number_input("Enter the upper x bound: ")
        y_lower = st.number_input("Enter the lower y bound: ")
        y_upper = st.number_input("Enter the upper y bound: ")
        st.pyplot(calculator.graph_3D((x_lower, x_upper), (y_lower, y_upper)))
    elif option == "Find a partial derivative of the function":
        var = st.text_input("Enter the variable to differentiate with respect to: ", "x")
        st.latex(calculator.differentiate(var))
    elif option == "Find gradient vector":
        st.latex(calculator.gradient_2D())
    elif option == "Find directional derivative":
        x_coord = st.number_input("Enter the x coordinate: ")
        y_coord = st.number_input("Enter the y coordinate: ")
        x_vector = st.number_input("Enter the x coordinate of the direction vector: ")
        y_vector = st.number_input("Enter the y coordinate of the direction vector: ")
        st.latex(calculator.directional_derivative((x_coord, y_coord), (x_vector, y_vector)))
    elif option == "Evaluate the double integral of the function over a 2-D region":
        x_lower = st.number_input("Enter the lower x limit of integration: ")
        x_upper = st.number_input("Enter the upper x limit of integration: ")
        y_lower = st.number_input("Enter the lower y limit of integration: ")
        y_upper = st.number_input("Enter the upper y limit of integration: ")
        order = st.text_input("Enter the order of integration: ", "dxdy")
        st.latex(calculator.integrate_2D((x_lower, x_upper), (y_lower, y_upper), order))
    elif option == "Evaluate the scalar line integral of the function in 2-space":
        x_curve = st.text_input("Enter the x component of the parameterized curve in terms of t: ", "0")
        y_curve = st.text_input("Enter the y component of the parameterized curve in terms of t: ", "0")
        t_lower = st.number_input("Enter the lower t bound: ")
        t_upper = st.number_input("Enter the upper t bound: ")
        st.latex(calculator.line_integral_2D(x_curve, y_curve, (t_lower, t_upper)))
else:      
    option = st.radio("Select an option: ", ["Evaluate function at a point",
                                "Find a partial derivative of the function", 
                                "Find gradient vector", "Find directional derivative", 
                                "Evaluate the triple integral of the function over 3-D region", 
                                "Evaluate the scalar line integral of the function in 3-space"])
    if option == "Evaluate function at a point":
        x_coord = st.number_input("Enter the x coordinate: ")
        y_coord = st.number_input("Enter the y coordinate: ")
        z_coord = st.number_input("Enter the z coordinate: ")
        st.latex(calculator.evaluate_at_2D_or_3D_point((x_coord, y_coord, z_coord)))
    elif option == "Find a partial derivative of the function":
        var = st.text_input("Enter the variable to differentiate with respect to: ", "x")
        st.latex(calculator.differentiate(var))
    elif option == "Find gradient vector":
        st.latex(calculator.gradient_3D())
    elif option == "Find directional derivative":
        x_coord = st.number_input("Enter the x coordinate: ")
        y_coord = st.number_input("Enter the y coordinate: ")
        z_coord = st.number_input("Enter the z coordinate: ")
        x_vector = st.number_input("Enter the x coordinate of the direction vector: ")
        y_vector = st.number_input("Enter the y coordinate of the direction vector: ")
        z_vector = st.number_input("Enter the z coordinate of the direction vector: ")
        st.latex(calculator.directional_derivative((x_coord, y_coord, z_coord), 
                                                   (x_vector, y_vector, z_vector)))
    elif option == "Evaluate the triple integral of the function over 3-D region":
        x_lower = st.number_input("Enter the lower x limit of integration: ")
        x_upper = st.number_input("Enter the upper x limit of integration: ")
        y_lower = st.number_input("Enter the lower y limit of integration: ")
        y_upper = st.number_input("Enter the upper y limit of integration: ")
        z_lower = st.number_input("Enter the lower z limit of integration: ")
        z_upper = st.number_input("Enter the upper z limit of integration: ")
        order = st.text_input("Enter the order of integration: ", "dxdydz")
        st.latex(calculator.integrate_3D((x_lower, x_upper), (y_lower, y_upper), 
                                         (z_lower, z_upper), order))
    elif option == "Evaluate the scalar line integral of the function in 3-space":
        x_curve = st.text_input("Enter the x component of the parameterized curve in terms of t: ", "0")
        y_curve = st.text_input("Enter the y component of the parameterized curve in terms of t: ", "0")
        z_curve = st.text_input("Enter the z component of the parameterized curve in terms of t: ", "0")
        t_lower = st.number_input("Enter the lower t bound: ")
        t_upper = st.number_input("Enter the upper t bound: ")
        st.latex(calculator.line_integral_3D(x_curve, y_curve, z_curve, (t_lower, t_upper)))

